import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import tifffile
import natsort
import h5py
from tqdm import tqdm
import torchvision
import imageio
import torchstain
from torchvision import transforms
import cv2

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import torch.nn.functional as F

from .utils import load_image

from stainlib.augmentation.augmenter import HedLighterColorAugmenter


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


def get_region_spacing(size, mode, divisions_fold):

    div_a = int(round(divisions_fold[0] * size))
    div_b = int(round(divisions_fold[1] * size))

    wp_predict = np.arange(div_a, div_b)

    if mode == "train":
        # remove val points to get train points
        wp_train = np.arange(size)
        mask = np.isin(wp_train, wp_predict, invert=True)
        wp_train = wp_train[mask]
        return wp_train
    else:
        return wp_predict


def find_patch_coordinates(w1, w2, patch_width=256, overlap=30):
    coordinates = []
    step_size = patch_width - overlap
    current_coord = w1

    while current_coord < w2:
        coordinates.append(min(current_coord, w2 - patch_width))
        current_coord += step_size

    return coordinates


def get_input_data(
    fp_nuc_seg,
    fp_hist,
    fp_nuc_sizes,
    mode,
    opts_data,
    fold_id,
    hsize,
    wsize,
    overlap,
    gene_names,
    divisions_fold,
    fp_expr,
    fp_cell_type,
    cell_types,
    experiment_path,
    demo_predict
):

    # cell gene expressions
    if fp_expr is not None:
        df_expr = pd.read_csv(fp_expr, index_col=0)
        df_expr = df_expr[gene_names]

    if fp_cell_type is not None:
        df_ct = pd.read_csv(fp_cell_type, index_col="c_id")
        is_all_numbers = pd.to_numeric(df_ct["ct"], errors="coerce").notna().all()
        # map to integers if ct is string
        if not is_all_numbers:
            ct_dict = dict(zip(cell_types, list(range(len(cell_types)))))
            df_ct["ct"] = df_ct["ct"].map(ct_dict).astype(int)
            # add 1 so that background = 0
            print(f"Cell type data shape, {df_ct.shape}")
        df_ct["ct"] = df_ct["ct"] + 1

    nuclei = load_image(fp_nuc_seg)
    hist = load_image(fp_hist)

    # nuclei = nuclei[2000:4000, 2000:4000]
    # hist = hist[2000:4000, 2000:4000]

    whole_h = hist.shape[0]
    whole_w = hist.shape[1]

    print(f"Histology image {hist.shape}, Nuclei {nuclei.shape}")

    # valid region of whole image
    wp = get_region_spacing(whole_h, mode, divisions_fold)
    nuclei_fold = nuclei[wp, :]

    # final cells = those in segmentation, meets min size, expr data, cell type data
    ids_seg = np.unique(nuclei_fold)
    ids_seg = ids_seg[ids_seg != 0]

    # meets min size req
    if fp_nuc_sizes is not False:
        df_sizes = pd.read_csv(fp_nuc_sizes, index_col=0)
        min_nuc_size = opts_data.min_nuc_area
        df_sizes = df_sizes[df_sizes["size_pix_histology"] >= min_nuc_size]

        ids_meet_min = df_sizes.index.tolist()

        all_intersect = list(set(ids_seg) & set(list(ids_meet_min)))
    else:
        all_intersect = list(set(ids_seg))

    if fp_expr is not None:
        all_intersect = list(set(all_intersect) & set(df_expr.index.tolist()))
        # get expr of the cells
        df_expr = df_expr[df_expr.index.isin(all_intersect)]
        assert list(df_expr.index) == df_expr.index.tolist()
        df_expr = opts_data.expr_scale * np.log1p(df_expr)
    else:
        df_expr = None

    if fp_cell_type is not None:
        all_intersect = list(set(all_intersect) & set(list(df_ct.index)))
        df_ct = df_ct.loc[all_intersect, :]
    else:
        df_ct = None

    all_intersect = natsort.natsorted(all_intersect)

    n_cells = len(all_intersect)
    print(f"{n_cells} cells")

    # overlapping patches
    w_starts = list(np.arange(0, whole_w - wsize, wsize - overlap))
    w_starts.append(whole_w - wsize)

    coord_idx = find_patch_coordinates(0, len(wp), patch_width=hsize, overlap=overlap)
    h_starts = wp[coord_idx]
    print("Patches min/max coords", h_starts.min(), h_starts.max() + hsize)

    # check there are cells in the patches
    print("Getting valid patches")
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    coords_starts_valid = []

    for hs, ws in tqdm(coords_starts):
        nuclei_p = nuclei[hs : hs + hsize, ws : ws + wsize]

        ids_seg = np.unique(nuclei_p)
        ids_seg = ids_seg[ids_seg != 0]
        valid_ids = list(set(ids_seg) & set(all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_p)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_p == k] = v

        if np.sum(nuclei_valid) > 0:
            coords_starts_valid.append((hs, ws))

    min_hs, min_ws = coords_starts_valid[0]
    max_hs, max_ws = coords_starts_valid[0]

    for hs, ws in coords_starts_valid:
        if hs < min_hs:
            min_hs = hs
        if hs > max_hs:
            max_hs = hs
        if ws < min_ws:
            min_ws = ws
        if ws > max_ws:
            max_ws = ws


    print("Standardisation")

    if demo_predict:
        fp_norms = f"{experiment_path}/standardisation_hist_demo_predict.npy"
    else:
        fp_norms = f"{experiment_path}/standardisation_hist_fold_{fold_id}.npy"

    if mode == "train":
        if not os.path.exists(fp_norms):
            hist_means = np.zeros(3)
            hist_stds = np.zeros(3)
            for hs, ws in tqdm(coords_starts):

                hist_p = hist[hs : hs + hsize, ws : ws + wsize]

                hist_means += np.mean(hist_p, (0, 1))
                hist_stds += np.std(hist_p, (0, 1))

            hist_means = hist_means / len(coords_starts)
            hist_stds = hist_stds / len(coords_starts)

            norms_hist = np.vstack((hist_means, hist_stds))
            np.save(fp_norms, norms_hist)

    norms_hist = np.load(fp_norms)

    return coords_starts_valid, hist, nuclei, all_intersect, df_ct, df_expr, norms_hist


class DataProcessing(data.Dataset):
    def __init__(
        self,
        opts_data_sources,
        opts_data,
        opts_regions,
        opts_comps,
        classes,
        gene_names,
        device,
        experiment_path,
        stain_aug,
        fold_id=1,
        mode="train",
        demo_predict=False
    ):

        # check all the files to load
        check_path(opts_data_sources.fp_nuc_seg)
        check_path(opts_data_sources.fp_hist)
        if opts_data_sources.fp_nuc_sizes is not False:
            check_path(opts_data_sources.fp_nuc_sizes)

        if mode != "predict":
            check_path(opts_data_sources.fp_expr)
            fp_expr = opts_data_sources.fp_expr
        else:
            fp_expr = None

        if getattr(opts_comps, "celltype", False) and mode != "predict":
            check_path(opts_data_sources.fp_cell_type)
            fp_cell_type = opts_data_sources.fp_cell_type
            self.cell_types = opts_data.cell_types
            self.comps_celltype = True
        else:
            fp_cell_type = None
            self.cell_types = None
            self.comps_celltype = False

        self.classes = classes
        self.mode = mode
        self.fold_id = fold_id
        self.gene_names = gene_names
        self.max_cells_per_patch = opts_data.max_cells_per_patch
        self.hsize = opts_data.hsize
        self.wsize = opts_data.wsize
        self.device = device
        self.experiment_path = experiment_path
        self.stain_aug = stain_aug
        self.demo_predict = demo_predict

        # fraction of image size: height start to height end
        divisions_fold = opts_regions.divisions[self.fold_id - 1]

        # overlap between tiles (pixels)
        if mode == "train":
            overlap = 0
        else:
            overlap = opts_data.overlap

        (
            coords_starts_valid,
            self.hist,
            self.nuclei,
            self.all_intersect,
            self.df_ct,
            self.df_expr,
            norms_hist,
        ) = get_input_data(
            opts_data_sources.fp_nuc_seg,
            opts_data_sources.fp_hist,
            opts_data_sources.fp_nuc_sizes,
            self.mode,
            opts_data,
            fold_id,
            self.hsize,
            self.wsize,
            overlap,
            gene_names,
            divisions_fold,
            fp_expr,
            fp_cell_type,
            self.cell_types,
            experiment_path,
            demo_predict
        )

        self.norms_hist = norms_hist.copy()
        self.coords_starts = coords_starts_valid

        self.n_patches = len(self.coords_starts)

        # Augmentation
        self.tfs = v2.Compose(
            [
                v2.ToImage(),
                # v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomApply([v2.RandomRotation((90, 90))], p=0.25),
                v2.RandomApply([v2.RandomRotation((180, 180))], p=0.25),
                v2.RandomApply([v2.RandomRotation((270, 270))], p=0.25),
                v2.ToDtype(torch.float32),
            ]
        )

        self.tfs_predict = v2.Compose(
            [
                v2.ToImage(),
                # v2.ToDtype(torch.float32, scale=True),
                v2.ToDtype(torch.float32),
            ]
        )

        self.hed_lighter_aug = HedLighterColorAugmenter()

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_patches

    def __getitem__(self, index):
        "Generates one sample of data"

        hs, ws = self.coords_starts[index]

        nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
        hist_patch = self.hist[hs : hs + self.hsize, ws : ws + self.wsize]

        if self.mode == "train" and self.stain_aug:
            # https://github.com/sebastianffx/stainlib/blob/main/stainlib_augmentation.ipynb
            self.hed_lighter_aug.randomize()
            hist_patch = self.hed_lighter_aug.transform(hist_patch)

        ids_seg = np.unique(nuclei_patch)
        ids_seg = ids_seg[ids_seg != 0]

        # make sure cells have valid data
        valid_ids = list(set(ids_seg) & set(self.all_intersect))
        invalid_ids = list(set(ids_seg) - set(valid_ids))
        dictionary = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_patch)
        for k, v in dictionary.items():
            nuclei_valid[nuclei_patch == k] = v

        if self.comps_celltype and self.mode != "predict":
            # map to cell type nuclei map
            dictionary = dict(zip(valid_ids, self.df_ct.loc[valid_ids, "ct"].tolist()))
            types_patch = np.copy(nuclei_valid)
            for k, v in dictionary.items():
                types_patch[nuclei_valid == k] = v
        else:
            types_patch = np.where(nuclei_valid > 0, 1, 0)

        # standardisation
        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        hist_patch = hist_patch - means
        hist_patch = hist_patch / stds

        patch_ids = np.unique(nuclei_valid)
        patch_ids = patch_ids[patch_ids != 0]

        n_cells = len(patch_ids)

        max_cells_per_patch = self.max_cells_per_patch
        if n_cells > max_cells_per_patch:
            print(
                "exceeds max_cells_per_patch cells, try increasing the value in config"
            )

        expr_pad = np.zeros((max_cells_per_patch, len(self.gene_names)))
        if self.mode != "predict":
            expr = self.df_expr.loc[patch_ids, :].to_numpy()
            expr_pad[:n_cells, :] = expr.copy()

        if self.mode != "predict":
            patch_expr = expr_pad[:n_cells, :].sum(axis=0).astype(np.float32)
        else:
            patch_expr = np.zeros((len(self.gene_names),), dtype=np.float32)

        gt_types_pad = np.zeros(max_cells_per_patch)
        if self.comps_celltype and self.mode != "predict":
            gt_types_pad[:n_cells] = self.df_ct.loc[patch_ids, "ct"].to_numpy() - 1
        gt_types_torch = torch.from_numpy(gt_types_pad).long()

        patch_ids_pad = np.zeros(max_cells_per_patch)
        patch_ids_pad[:n_cells] = patch_ids.copy()
        patch_ids_torch = torch.from_numpy(patch_ids_pad).long()

        n_cells = np.array([n_cells])
        n_cells_torch = torch.from_numpy(n_cells).long()

        x_input = np.concatenate(
            (
                np.expand_dims(nuclei_valid, -1),
                np.expand_dims(types_patch, -1),
                hist_patch,
            ),
            -1,
        )

        if self.mode == "train":
            x_input = self.tfs(x_input)
        else:
            x_input = self.tfs_predict(x_input)

        nuclei_torch = x_input[0, :, :].type(torch.LongTensor)
        types_patch_torch = x_input[1, :, :].type(torch.LongTensor)
        hist_torch = x_input[2:, :, :]

        expr_torch = torch.from_numpy(expr_pad).float()
        patch_expr_torch = torch.from_numpy(patch_expr).float()

        return (
            nuclei_torch,
            types_patch_torch,
            hist_torch,
            expr_torch,
            n_cells_torch,
            gt_types_torch,
            patch_ids_torch,
            patch_expr_torch,
        )

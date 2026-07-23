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
from model.model_text_encoder import load_text_model

import loki.utils
import loki.preprocess


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


def get_region_spacing(size, mode, divisions_fold):
    div_a = int(round(divisions_fold[0] * size))
    div_b = int(round(divisions_fold[1] * size))

    wp_predict = np.arange(div_a, div_b)

    if mode == "train":
        wp_train = np.arange(size)
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

    if fp_expr is not None:
        df_expr = pd.read_csv(fp_expr, index_col=0)
        df_expr = df_expr[gene_names]

    if fp_cell_type is not None:
        df_ct = pd.read_csv(fp_cell_type, index_col="c_id")
        is_all_numbers = pd.to_numeric(df_ct["ct"], errors="coerce").notna().all()
        if not is_all_numbers:
            ct_dict = dict(zip(cell_types, list(range(len(cell_types)))))
            df_ct["ct"] = df_ct["ct"].map(ct_dict).astype(int)
            print(f"Cell type data shape, {df_ct.shape}")
        df_ct["ct"] = df_ct["ct"] + 1

    nuclei = load_image(fp_nuc_seg)
    hist = load_image(fp_hist)


    whole_h = hist.shape[0]
    whole_w = hist.shape[1]

    print(f"Histology image {hist.shape}, Nuclei {nuclei.shape}")

    wp = get_region_spacing(whole_h, mode, divisions_fold)
    nuclei_fold = nuclei[wp, :]

    ids_seg = np.unique(nuclei_fold)
    ids_seg = ids_seg[ids_seg != 0]

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

    w_starts = list(np.arange(0, whole_w - wsize, wsize - overlap))
    w_starts.append(whole_w - wsize)

    coord_idx = find_patch_coordinates(0, len(wp), patch_width=hsize, overlap=overlap)
    h_starts = wp[coord_idx]

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
        demo_predict=False,
        text_model_path=None,
    ):
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
        else:
            fp_cell_type = None

        self.mode = mode
        self.fold_id = fold_id
        self.gene_names = list(gene_names)
        self.device = device
        self.experiment_path = experiment_path
        self.stain_aug = stain_aug
        self.demo_predict = demo_predict

        divisions_fold = opts_regions.divisions[self.fold_id - 1]

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
            opts_data.hsize,
            opts_data.wsize,
            overlap,
            self.gene_names,
            divisions_fold,
            fp_expr,
            fp_cell_type,
            None,
            experiment_path,
            demo_predict,
        )

        self.norms_hist = norms_hist.copy()
        self.coords_starts = coords_starts_valid
        self.hsize = opts_data.hsize
        self.wsize = opts_data.wsize
        self.max_cells_per_patch = opts_data.max_cells_per_patch

        self.tfs = v2.Compose(
            [
                v2.ToImage(),
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
                v2.ToDtype(torch.float32),
            ]
        )

        self.text_model_base, _, self.tokenizer = load_text_model(text_model_path, torch.device("cpu"))


    def __len__(self):
        return len(self.coords_starts)

    def __getitem__(self, index):
        hs, ws = self.coords_starts[index]

        nuclei_patch = self.nuclei[hs : hs + self.hsize, ws : ws + self.wsize]
        hist_patch = self.hist[hs : hs + self.hsize, ws : ws + self.wsize]

        if self.mode == "train" and self.stain_aug:
            try:
                self.hed_lighter_aug.randomize()
                hist_patch = self.hed_lighter_aug.transform(hist_patch)
            except Exception:
                pass

        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        hist_patch = hist_patch - means
        hist_patch = hist_patch / stds

        patch_ids = np.unique(nuclei_patch)
        patch_ids = patch_ids[patch_ids != 0]
        n_cells = len(patch_ids)
        max_cells_per_patch = self.max_cells_per_patch

        expr_pad = np.zeros(
            (max_cells_per_patch, len(self.gene_names)), dtype=np.float32
        )
        if self.mode != "predict" and self.df_expr is not None and n_cells > 0:
            valid_ids = [pid for pid in patch_ids if pid in self.df_expr.index]
            if len(valid_ids) > 0:
                expr = self.df_expr.loc[valid_ids, :].to_numpy(dtype=np.float32)
                expr_pad[: len(valid_ids), :] = expr.copy()
        if n_cells > 0:
            patch_expr_all = expr_pad[:n_cells, :].sum(axis=0).astype(np.float32)
        else:
            patch_expr_all = expr_pad.sum(axis=0).astype(np.float32)

        patch_expr = patch_expr_all.astype(np.float32)

        patch_expr_torch = torch.from_numpy(patch_expr).float()

        k = min(50, len(self.gene_names))
        if k > 0:
            topk_idx = np.argsort(-patch_expr)[:k]
            top_genes = [self.gene_names[i] for i in topk_idx]
            patch_expr_text = " ".join(top_genes)
        else:
            patch_expr_text = ""

        if self.mode == "train":
            x = self.tfs(hist_patch)
        else:
            x = self.tfs_predict(hist_patch)

        if isinstance(x, torch.Tensor):
            hist_torch = x
        else:
            x_arr = np.asarray(x).astype(np.float32)
            if x_arr.ndim == 2:
                x_arr = np.expand_dims(x_arr, -1)
            hist_torch = torch.from_numpy(x_arr.transpose(2, 0, 1)).float()

        embedding_text = loki.utils.encode_texts(
            self.text_model_base, self.tokenizer, [patch_expr_text], device="cpu"
        ) 
        return (
            hist_torch,
            patch_expr_torch,
            embedding_text,
        )


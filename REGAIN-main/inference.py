import argparse
import logging
import os
import sys
import natsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

from model.adjustments import *
from dataio.dataset_input import DataProcessing
from model.model import Framework
from utils.utils import *

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import csv

from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

from sklearn.metrics import accuracy_score


def main(config):
    opts = json_file_to_pyobj(config.config_file)

    if os.path.basename(config.config_file) == "config_demo.json" and config.mode == "predict":
        demo_predict = True
    else:
        demo_predict = False

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(config.gpu_id)

    make_new = False
    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    model_dir = experiment_path + "/" + opts.experiment_dirs.model_dir
    if config.mode == "predict":
        predict_output_dir = (
            experiment_path + "/" + opts.experiment_dirs.predict_output_dir + "/"
        )
    else:
        predict_output_dir = (
            experiment_path + "/" + opts.experiment_dirs.val_output_dir + "/"
        )
    os.makedirs(predict_output_dir, exist_ok=True)

    logging.info("Initialising model")

    use_avgexp = False
    use_celltype = getattr(opts.comps, "celltype", False)
    use_neighb = False

    if use_celltype:
        classes = opts.data.cell_types
        n_classes = len(classes)
        print(classes)
        print(f"Num cell types {n_classes}")
    else:
        n_classes = 0
        classes = []

    fp_genes = os.path.join(experiment_path, "genes.txt")
    gene_names = read_txt(fp_genes)
    n_genes = len(gene_names)
    print(f"{n_genes} genes")

    n_ref = 0
    expr_ref_torch = None

    model = Framework(
        n_classes,
        n_genes,
        opts.model.emb_dim,
        device,
        n_ref,
        use_avgexp,
        use_celltype,
        use_neighb,
    )

    if demo_predict:
        saved_model_epochs = ["demo_predict"]
    elif config.epoch in ["last", "all"]:
        saved_model_paths = glob.glob(f"{model_dir}/epoch_*.pth")
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_names = [
            (os.path.basename(x)).split(".")[0] for x in saved_model_paths
        ]
        saved_model_epochs = [x.split("_")[1] for x in saved_model_names]
        saved_model_epochs = list(set(saved_model_epochs))
        saved_model_epochs = sorted_alphanumeric(saved_model_epochs)
        if config.epoch == "all":
            saved_model_epochs = np.array(saved_model_epochs, dtype="int")
        elif config.epoch == "last":
            saved_model_epochs = np.array(saved_model_epochs[-1], dtype="int")
            saved_model_epochs = [saved_model_epochs]
    else:
        epoch_valid = is_valid_positive_int(config.epoch)
        if epoch_valid:
            saved_model_epochs = [int(config.epoch)]
        else:
            sys.exit("Invalid --epoch: specify an integer, or use 'last' for the most recent, or 'all' for all epochs")

    logging.info("Preparing data")

    if config.mode == "val":
        opts_data_sources = opts.data_sources_train_val
    elif config.mode == "predict":
        opts_data_sources = opts.data_sources_predict
    else:
        sys.exit("Invalid --mode: choose either val or predict")

    if config.mode == "val":
        opts_regions = opts.regions_val
    else:
        opts_regions = opts.regions_predict

    predict_dataset = DataProcessing(
        opts_data_sources,
        opts.data,
        opts_regions,
        opts.comps,
        classes,
        gene_names,
        device,
        experiment_path,
        False,
        config.fold_id,
        mode=config.mode,
        demo_predict=demo_predict
    )
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=opts.training.batch_size,
        shuffle=False,
        num_workers=opts.data.num_workers,
        drop_last=False,
    )

    n_predict_examples = len(dataloader)
    logging.info("Total number of patches: %d" % n_predict_examples)

    logging.info("Begin prediction")

    all_f1_ct = []
    all_corr_mean = []

    with torch.no_grad():

        for epoch_idx, predict_epoch in enumerate(saved_model_epochs):

            if demo_predict:
                load_path = f"{model_dir}/model.pth"
            else:
                load_path = model_dir + "/epoch_%d_model.pth" % (predict_epoch)

            checkpoint = torch.load(load_path, map_location="cpu")

            model.load_state_dict(checkpoint["model_state_dict"])
            print("Predict using " + load_path)

            model.to(device)

            model = model.eval()

            pbar = tqdm(dataloader)

            all_gt_ct = []
            all_pr_ct = []
            all_pr_adj = []
            all_ids = []
            all_expr = None
            all_expr_gt = None
            all_comp_est = np.zeros(
                (n_predict_examples * opts.training.batch_size, n_classes)
            )
            all_comp_est_i = 0
            all_comp_est_by_cell = None
            all_area = None

            all_embedding = None

            all_error = None

            for (
                batch_nuclei,
                _,
                batch_he_img,
                batch_expr,  # val
                batch_n_cells,
                batch_ct,  # val and use_celltype
                patch_ids,
                batch_patch_expr,
            ) in pbar:

                batch_nuclei = batch_nuclei.to(device)
                batch_he_img = batch_he_img.to(device)
                batch_expr = batch_expr.to(device)  # all zeros placeholder if predict
                batch_n_cells = batch_n_cells.to(device)
                batch_ct = batch_ct.to(device)  # all zeros placeholder if predict
                patch_ids = patch_ids.to(device)
                batch_patch_expr = batch_patch_expr.to(device)

                # assume no GT available, don't calculate performance
                if config.mode == "predict":
                    batch_ct_input = None
                    batch_expr_input = None
                else:
                    batch_ct_input = batch_ct.clone()
                    batch_expr_input = batch_expr.clone()

                (
                    out_cell_type,
                    _,
                    batch_ct_pc,  # all zeros placeholder if predict
                    out_expr,
                    out_expr_immune,
                    out_expr_invasive,
                    _,
                    _,
                    _,
                    _,
                    batch_expr_pc,  # all zeros placeholder if predict
                    comp_estimated,
                    batch_area,
                    patch_ids_pc,
                    embeddings,
                    patch_pred,
                ) = model(
                    batch_he_img,
                    batch_nuclei,
                    batch_n_cells,
                    expr_ref_torch,
                    batch_ct_input,
                    batch_expr_input,
                    patch_ids=patch_ids,
                )

                if out_expr.shape[0] == 0:
                    continue

                if use_neighb:
                    # neighbourhood compositions
                    comp_out_raw = torch.nn.functional.softmax(out_cell_type, dim=1)
                    comp_out_raw = torch.argmax(comp_out_raw, 1)
                    comp_out = torch.nn.functional.one_hot(
                        comp_out_raw, num_classes=n_classes
                    )
                    comp_out = comp_out.float()
                    comp_out = torch.mean(comp_out, 0)

                    # adjustments based on predicted composition
                    adjusted_out_cell_type = []
                    comp_estimated_sum = torch.zeros(n_classes).to(device)

                    for i_batch in range(batch_n_cells.shape[0]):
                        n_cells_batch = int(batch_n_cells[i_batch])

                        if n_cells_batch > 0:
                            idx_start = torch.sum(batch_n_cells[:i_batch]).item()
                            idx_end = idx_start + n_cells_batch

                            comp_out_raw_patch = torch.nn.functional.softmax(
                                out_cell_type[idx_start:idx_end, :], dim=1
                            )
                            comp_out_raw_patch = torch.argmax(comp_out_raw_patch, 1)
                            comp_out_patch = torch.nn.functional.one_hot(
                                comp_out_raw_patch, num_classes=n_classes
                            )
                            comp_out_patch = comp_out_patch.float()
                            comp_out_patch = torch.mean(comp_out_patch, 0)

                            # refinements based on predicted compositions

                            # mask highly confident cell predictions
                            pred_logits = F.softmax(
                                out_cell_type[idx_start:idx_end, :], dim=1
                            )

                            ct_index_imm = [opts.data.cell_types.index("T")]
                            high_conf_imm = torch.where(
                                pred_logits[:, ct_index_imm] > opts.data.high_conf_prob
                            )[0]
                            high_conf_imm = high_conf_imm.cpu().numpy().tolist()

                            # cell type refinement

                            (
                                adjusted_out_cell_type_patch_immune,
                                idx_swapped_immune_all,
                                idx_swapped_immune,
                            ) = adjust_pr(
                                out_cell_type[idx_start:idx_end, :],
                                comp_estimated[i_batch, :],
                                comp_out_patch,
                                opts.data.cell_types,
                                ["B", "Myeloid", "T"],
                                ignore_idx=high_conf_imm,
                                scale=config.alpha,
                            )

                            # ensure consistency with expressions
                            if len(idx_swapped_immune) > 0:
                                idx_swapped_immune = [
                                    x + idx_start for x in idx_swapped_immune
                                ]
                                out_expr[idx_swapped_immune] = out_expr_immune[
                                    idx_swapped_immune
                                ]

                            if config.is_invasive:

                                # malignant
                                (
                                    adjusted_out_cell_type_patch_invasive,
                                    idx_swapped_invasive_all,
                                    idx_swapped_invasive,
                                ) = adjust_pr(
                                    out_cell_type[idx_start:idx_end, :],
                                    comp_estimated[i_batch, :],
                                    comp_out_patch,
                                    opts.data.cell_types,
                                    ["Malignant", "Epithelial"],
                                    ignore_idx=[],
                                    scale=10000,
                                )

                                if len(idx_swapped_invasive) > 0:
                                    idx_swapped_invasive = [
                                        x + idx_start for x in idx_swapped_invasive
                                    ]
                                    out_expr[idx_swapped_invasive] = out_expr_invasive[
                                        idx_swapped_invasive
                                    ]

                            adjusted_out_cell_type_patch = (
                                adjusted_out_cell_type_patch_immune.copy()
                            )
                            if config.is_invasive:
                                for index in idx_swapped_invasive_all:
                                    adjusted_out_cell_type_patch[index] = (
                                        adjusted_out_cell_type_patch_invasive[index]
                                    )

                            adjusted_out_cell_type.extend(adjusted_out_cell_type_patch)

                            # neighbourhood compositions
                            comp_estimated_sum += n_cells_batch * comp_estimated[i_batch, :]

                            all_comp_est[all_comp_est_i, :] = (
                                n_cells_batch
                                * comp_estimated[i_batch, :].detach().cpu().numpy()
                            )

                            repeated_by_cells = np.tile(
                                all_comp_est[all_comp_est_i, :], (n_cells_batch, 1)
                            )

                            # also for each corresponding cell
                            if all_comp_est_by_cell is None:
                                all_comp_est_by_cell = repeated_by_cells.copy()
                            else:
                                all_comp_est_by_cell = np.vstack(
                                    (all_comp_est_by_cell, repeated_by_cells)
                                )

                            all_comp_est_i += 1

                    all_pr_adj.extend(adjusted_out_cell_type)


                out_expr = np.expm1(out_expr.detach().cpu().numpy() / opts.data.expr_scale)
                out_expr_gt = np.expm1(batch_expr_pc.detach().cpu().numpy() / opts.data.expr_scale)

                error = np.abs(out_expr - out_expr_gt) / (np.abs(out_expr) + np.abs(out_expr_gt) + 1e-6)
                if  all_error is None:
                    all_error = error.copy()
                else:
                    all_error = np.vstack((all_error, error))

                if all_expr is None:
                    all_expr = out_expr.copy()
                    all_expr_gt = out_expr_gt.copy()
                else:
                    all_expr = np.vstack((all_expr, out_expr))
                    all_expr_gt = np.vstack((all_expr_gt, out_expr_gt))


                embeddings = embeddings.detach().cpu().numpy()
                if all_embedding is None:
                    all_embedding = embeddings.copy()
                else:
                    all_embedding = np.vstack((all_embedding, embeddings))


                if all_area is None:
                    all_area = batch_area.detach().cpu().numpy()
                else:
                    all_area = np.hstack((all_area, batch_area.detach().cpu().numpy()))

                patch_ids_pc = list(patch_ids_pc.detach().cpu().numpy())
                all_ids.extend(patch_ids_pc)

                if use_celltype:
                    batch_ct_pc = list(batch_ct_pc.detach().cpu().numpy())
                    out_cell_type_ct = torch.argmax(out_cell_type, dim=1)
                    out_cell_type_ct = list(out_cell_type_ct.detach().cpu().numpy())
                    all_gt_ct.extend(batch_ct_pc)
                    all_pr_ct.extend(out_cell_type_ct)

            combined = np.vstack((np.array(all_ids), all_area, np.arange(len(all_ids))))
            combined = np.transpose(combined)
            combined_df = pd.DataFrame(
                combined, index=np.arange(len(all_ids)), columns=["id", "area", "index"]
            )
            sorted_df = combined_df.sort_values(by="area", ascending=False)
            unique_df = sorted_df.drop_duplicates(subset="id", keep="first")

            unique_indices = unique_df["index"].astype(int).tolist()

            unique_indices = sorted(unique_indices)
            all_ids = [all_ids[ui] for ui in unique_indices]

            if use_celltype:
                all_gt_ct = [all_gt_ct[ui] for ui in unique_indices]
                all_pr_ct = [all_pr_ct[ui] for ui in unique_indices]
            if use_neighb:
                all_pr_adj = [all_pr_adj[ui] for ui in unique_indices]

            all_expr = all_expr[unique_indices, :]
            all_expr_gt = all_expr_gt[unique_indices, :]

            all_embedding = all_embedding[unique_indices, :]
            all_error = all_error[unique_indices, :]

            df_all_expr = pd.DataFrame(all_expr, index=all_ids, columns=gene_names)
            fp_expr_out = f"{predict_output_dir}/epoch_{predict_epoch}_expr.csv"
            df_all_expr.to_csv(fp_expr_out)
            print(
                f"Saved predicted expressions of {len(unique_indices)} cells to {fp_expr_out}"
            )

            df_all_expr_gt = pd.DataFrame(
                all_expr_gt, index=all_ids, columns=gene_names
            )

            df_all_expr_gt.to_csv(f"{predict_output_dir}/epoch_{predict_epoch}_expr_gt.csv")
            print(
                f"Saved ground truth expressions of {len(unique_indices)} cells to {predict_output_dir}/epoch_{predict_epoch}_expr_gt.csv"
            )

            df_error = pd.DataFrame(all_error, index=all_ids, columns=gene_names)
            fp_error_out = f"{predict_output_dir}/epoch_{predict_epoch}_error.csv"
            df_error.to_csv(fp_error_out)
            print(f"Saved error matrix to {fp_error_out}")


            df_all_embedding = pd.DataFrame(all_embedding, index=all_ids)
            fp_embedding_out = f"{predict_output_dir}/epoch_{predict_epoch}_embedding.csv"
            df_all_embedding.to_csv(fp_embedding_out)

            if config.mode == "val":

                print("***expression correlation***")
                all_rowwise = df_all_expr_gt.corrwith(df_all_expr, axis=0)
                mean_corr = np.nanmean(all_rowwise)
                print(f"PCC mean: {mean_corr}")
                all_corr_mean.append(mean_corr)

                if use_celltype:
                    f1 = f1_score(all_gt_ct, all_pr_ct, average=None)
                    f1_mean = np.mean(f1)
                    all_f1_ct.append(f1_mean)

                    acc = accuracy_score(all_gt_ct, all_pr_ct)
                    print(f"Cell Type Accuracy (weighted): {acc:.4f}")

    if config.mode == "val":
        print("***best epoch***")
        rank_corr = rankdata(-np.array(all_corr_mean), method="min")
        if use_celltype:
            rank_f1 = rankdata(-np.array(all_f1_ct), method="min")
            ranksums = rank_f1 + rank_corr
        else:
            ranksums = rank_corr.copy()
        best_idx = np.argmin(ranksums)
        best_epoch = saved_model_epochs[best_idx]
        print(f"PCC best epoch {best_epoch} mean {all_corr_mean[best_idx]}")


logging.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--epoch",
        default="all",
        type=str,
        help="epoch to run: specify an integer, or use 'last' for the most recent, or 'all' for all epochs",
    )
    parser.add_argument(
        "--mode",
        default="predict",
        type=str,
        help="predict or val",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    parser.add_argument(
        "--alpha",
        default=1,
        type=float,
        help="hyperparameter for neighbourhood composition refinement",
    )

    parser.add_argument(
        "--is_invasive", action="store_true", default=False, help="Enable invasive mode"
    )

    config = parser.parse_args()
    main(config)

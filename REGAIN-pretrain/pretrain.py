import logging
import shutil
import time
import torch

import argparse
import logging
import os
import sys
from pathlib import Path
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort

from dataio.dataset_input_for_single_panel import DataProcessing
from model.model_image_encoder import Image_Encoder
from model.model_text_encoder import load_text_model, Text_Encoder
from utils.utils import *

import loki.utils
import loki.preprocess

SCRIPT_DIR = Path(__file__).resolve().parent

def get_device(gpu_id):
    import torch, os
    if not torch.cuda.is_available():
        print("CUDA not available -> use CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_id is None or gpu_id < 0 or gpu_id >= n:
        print(f"Requested gpu_id={gpu_id} out of range (0..{n-1}) -> use CPU")
        return torch.device("cpu")
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using device: {device} (cuda count={n})")
    return device


def train_contrastive(
    dataloader: DataLoader,
    image_encoder: torch.nn.Module,
    text_model_2: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    save_path: str = None,
    log_interval: int = 10,
):
    image_encoder.to(device)
    image_encoder.train()

    text_model_2.to(device)
    text_model_2.train()
    for p in text_model_2.parameters():
        p.requires_grad = True

    params = list(image_encoder.parameters()) + [
        p for p in text_model_2.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
           
            hist_batch, _, embedding_text = batch          

            hist_batch = hist_batch.to(device)
            embedding_text = embedding_text.to(device)
            B = hist_batch.shape[0]
            if B == 0:
                continue

            optimizer.zero_grad()

            img_emb = image_encoder(hist_batch)
            if isinstance(img_emb, (tuple, list)):
                img_emb = img_emb[0]
            img_emb = img_emb.view(B, -1)
            img_emb = F.normalize(img_emb, p=2, dim=-1)

            text_emb = text_model_2(embedding_text)
            text_emb = text_emb.view(B, -1)
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            targets = torch.arange(B, device=device, dtype=torch.long)

            sim = torch.matmul(img_emb, text_emb.t()) / temperature
            loss_i2t = loss_fn(sim, targets)
            loss_t2i = loss_fn(sim.t(), targets)

            loss = 0.5 * (loss_i2t + loss_t2i)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if batch_idx % log_interval == 0:
                print(
                    f"[Epoch {epoch} Batch {batch_idx}] "
                    f"loss={loss.item():.4f}"
                )

        epoch_time = time.time() - t0
        avg_loss = running_loss / max(1, n_batches)
        print(
            f"Epoch {epoch} finished. time={epoch_time:.1f}s "
            f"avg_loss={avg_loss:.4f}"
        )

        if save_path is not None:
            save_dir = Path(save_path).expanduser().resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_fp = save_dir / f"epoch{epoch}.pth"
            tmp_fp = save_dir / f".epoch{epoch}.pth.tmp"
            torch.save(
                {
                    "image_encoder": image_encoder.state_dict(),
                    "text_model_2": text_model_2.state_dict(),
                    "epoch": epoch,
                },
                tmp_fp,
            )
            tmp_fp.replace(ckpt_fp)
            if not ckpt_fp.exists() or ckpt_fp.stat().st_size == 0:
                raise RuntimeError(f"Checkpoint save failed: {ckpt_fp}")
            print(f"Saved checkpoint: {ckpt_fp} ({ckpt_fp.stat().st_size / 1024 / 1024:.2f} MB)")

    return image_encoder, text_model_2


def main(config):

    os.chdir(SCRIPT_DIR)
    config_path = Path(config.config_file)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    config.config_file = str(config_path)
    opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(config.gpu_id)
    print("Using device:", device)

    make_new = True
    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = str(SCRIPT_DIR / "experiments" / timestamp)
    os.makedirs(os.path.join(experiment_path, opts.experiment_dirs.model_dir), exist_ok=True)

    shutil.copyfile(
        config.config_file,
        experiment_path + "/" + os.path.basename(config.config_file),
    )

    use_avgexp = False
    use_celltype = getattr(opts.comps, "celltype", False)

    if use_celltype:
        classes = opts.data.cell_types
        n_classes = len(classes)
        print(classes)
        print(f"Num cell types {n_classes}")
    else:
        classes = []
        n_classes = 0

    df_expr = pd.read_csv(opts.data_sources_train_val.fp_expr, index_col=0)
    gene_names = df_expr.columns.tolist()
    n_genes = len(gene_names)
    print(f"{n_genes} genes")
    fp_out = os.path.join(experiment_path, "genes.txt")
    with open(fp_out, "w") as f:
        for line in gene_names:
            f.write(f"{line}\n")
    logging.info("Preparing data for contrastive training")

    train_dataset = DataProcessing(
        opts_data_sources=opts.data_sources_train_val,
        opts_data=opts.data,
        opts_regions=opts.regions_val,
        opts_comps=opts.comps,
        classes=classes,
        gene_names=gene_names,
        device=device,
        experiment_path=experiment_path,
        stain_aug=opts.training.stain_aug,
        fold_id=config.fold_id,
        mode="train",
        demo_predict=False,
        text_model_path = opts.model.text_model_base_path
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=opts.training.batch_size,
        shuffle=True,
        num_workers=opts.data.num_workers,
        pin_memory=True,
    )

    emb_dim = opts.model.emb_dim
    image_encoder = Image_Encoder(
        n_classes=n_classes,
        emb_dim=emb_dim,
        device=device,
        use_celltype=use_celltype,
        in_channels=3,
    )

    text_model_2 = Text_Encoder(input_size=opts.model.text_base_dim, hidden_size=emb_dim)

    resume_epoch  = config.resume_epoch
    if resume_epoch and resume_epoch != 0:
        ckpt_dir = SCRIPT_DIR / "experiments" / getattr(opts.experiment_dirs, "load_dir", "")
        ckpt_fp = os.path.join(
            str(ckpt_dir),
            getattr(opts.experiment_dirs, "model_dir", ""),
            "contrastive_pretrain",
            f"epoch{resume_epoch}.pth",
        )
        if os.path.exists(ckpt_fp):
            ckpt = torch.load(ckpt_fp, map_location=device)
            image_encoder.load_state_dict(ckpt.get("image_encoder", {}), strict=False)
            text_model_2.load_state_dict(ckpt.get("text_model_2", {}), strict=False)
            print(f"Loaded resume checkpoint: {ckpt_fp}")
        else:
            print(f"Resume checkpoint not found: {ckpt_fp}")

    epochs = getattr(opts.training, "total_epochs", 10)
    lr = getattr(opts.training, "learning_rate", 1e-4)
    weight_decay = getattr(opts.training, "weight_decay", 1e-5)
    temperature = getattr(opts.training, "temperature", 0.07)

    save_path = os.path.join(
        experiment_path, opts.experiment_dirs.model_dir, "contrastive_pretrain"
    )

    train_contrastive(
        dataloader=dataloader,
        image_encoder=image_encoder,
        text_model_2=text_model_2,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        save_path=save_path,
    )


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="configs/config_demo.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=0,
        type=int,
        help="resume training from this epoch, set to 0 for new training",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--gpu_id",
        default=1,
        type=int,
        help="which GPU to use",
    )
    

    cfg = parser.parse_args()
    main(cfg)

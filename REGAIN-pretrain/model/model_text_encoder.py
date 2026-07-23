import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from PIL import Image

import loki.utils
import loki.preprocess
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
import torch.nn as nn


def load_text_model(model_path, device):


    model_name = 'coca_ViT-L-14'
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=False)  

    tokenizer = get_tokenizer(model_name)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('module.', ''): v for k,v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)


    model.to(device).eval()
    return model, preprocess, tokenizer


class Text_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Text_Encoder, self).__init__()

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .backbone import *


class Image_Encoder(nn.Module):

    def __init__(
        self,
        n_classes,
        emb_dim,
        device,
        use_celltype,
        in_channels=3,
    ):
        super(Image_Encoder, self).__init__()

        n_classes_backbone = n_classes + 1 if use_celltype else 2
        self.cnn = Backbone(
            n_channels=in_channels,
            bilinear=True,
            is_deconv=True,
            is_batchnorm=True,
            n_classes=n_classes_backbone,
        )

        dim_fv = 384 * 2
        self.hidden_size = emb_dim
        self.device = device
        self.embed_hist = Embed(dim_fv, self.hidden_size)

        
    def forward(
        self,
        x_hist,
    ):
        out_map, hd1, h1 = self.cnn(x_hist)

        patch_area_hd1 = hd1.shape[2] * hd1.shape[3]
        patch_area_h1 = h1.shape[2] * h1.shape[3]
        fv_hd1 = torch.sum(hd1, (2, 3)) / patch_area_hd1
        fv_h1 = torch.sum(h1, (2, 3)) / patch_area_h1

        batch_size = x_hist.shape[0]

        for i_batch in range(batch_size):
           
            if i_batch == 0:
                patch_image_fv = torch.cat((fv_hd1[i_batch], fv_h1[i_batch], fv_hd1[i_batch], fv_h1[i_batch]), 0)
                embeddings_image = self.embed_hist(patch_image_fv.unsqueeze(0))

            else:
                patch_image_fv = torch.cat((fv_hd1[i_batch], fv_h1[i_batch], fv_hd1[i_batch], fv_h1[i_batch]), 0)
                patch_emb = self.embed_hist(patch_image_fv.unsqueeze(0))
                embeddings_image = torch.cat((embeddings_image, patch_emb), 0)    

        return (
            embeddings_image,

        )

    



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from oformer.attention_module import LinearAttention, StandardAttention, FeedForward, PreNorm
from oformer.encoder_module import SpatialEncoder2D
from oformer.decoder_module import PointWiseDecoder2DSimple

    
class BlastOFormer(nn.Module):
    def __init__(self,
                 encoder_input_channels,
                 encoder_in_emb_dim,
                 encoder_out_seq_emb_dim,
                 encoder_heads,
                 encoder_depth,
                 encoder_res,
                 decoder_latent_channels,
                 decoder_out_channels,
                 decoder_res,
                 decoder_scale):
        super().__init__()
        self.encoder = SpatialEncoder2D(encoder_input_channels, encoder_in_emb_dim, encoder_out_seq_emb_dim, encoder_heads, encoder_depth, encoder_res)
        self.decoder = PointWiseDecoder2DSimple(decoder_latent_channels, decoder_out_channels, decoder_res, decoder_scale)

    def forward(self, x, pos):
        z = self.encoder(x, pos)
        x = self.decoder(z, pos, pos)
        return x
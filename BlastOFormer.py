import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from oformer.attention_module import LinearAttention, StandardAttention, FeedForward, PreNorm
from oformer.encoder_module import SpatialEncoder2D
from oformer.decoder_module import PointWiseDecoder2DSimple

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            Rearrange("b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, out_channels)
        )
    def forward(self, x):
        return self.proj(x)
    
class UnpatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.h, self.w = img_size // patch_size, img_size // patch_size
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels),
            Rearrange("b (h w) (p1 p2 c) -> b (h p1) (w p2) c", 
                      p1=patch_size, p2=patch_size, h=self.h, w=self.w)
        )
    def forward(self, x):
        return self.proj(x)
    
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
                 decoder_scale,
                input_channels,
                 patch_size,
                 img_size):
        super().__init__()
        self.encoder = SpatialEncoder2D(encoder_input_channels, encoder_in_emb_dim, encoder_out_seq_emb_dim, encoder_heads, encoder_depth, encoder_res)
        self.decoder = PointWiseDecoder2DSimple(decoder_latent_channels, decoder_out_channels, decoder_res, decoder_scale)
        self.patch_embed = PatchEmbed(input_channels, encoder_input_channels, patch_size)
        self.position_patch_embed = PatchEmbed(2, 2, patch_size)
        self.unpatch_embed = UnpatchEmbed(decoder_out_channels, decoder_out_channels, patch_size, img_size)

    def forward(self, x, pos):
        print(f'x shape in forward:{x.shape}')
        print(f'pos shape in forward: {pos.shape}')
        x = self.patch_embed(x)
        print(f'x shape after patch_embed: {x.shape}')
        pos = self.position_patch_embed(pos)
        print(f'pos shape after position_patch_embed: {pos.shape}')
        z = self.encoder(x, pos)
        print(f'z shape after encoder: {z.shape}')
        x = self.decoder(z, pos, pos)
        print(f'x shape after decoder: {x.shape}')
        x = self.unpatch_embed(x)
        print(f'x shape after unpatch_embed: {x.shape}')
        return x
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.modified_linear import *

# helpers
def make_tuple(t):
    """
    return the input if it's already a tuple.
    return a tuple of the input if the input is not already a tuple.
    """
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=7, dim_head=64, dropout=0.):
        """
        reduced the default number of heads by 1 per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EarlyConvViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        3x3 conv, stride 1, 5 conv layers per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()

        n_filter_list = (channels, 48, 96, 192, 384)  # hardcoding for now because that's what the paper used

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=dim, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.fc = CosineLinear(dim, num_classes)

    def get_output_dim(self):
        return self.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):

        if second_iter:
            in_features = self.fc.in_features
            out_features1 = self.fc.fc1.out_features
            out_features2 = self.fc.fc2.out_features
            #print("in_features:", in_features, "out_features1:", \
            #    out_features1, "out_features2:", out_features2)
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, out_features2)
            new_fc.fc1.weight.data[:out_features1] = self.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.fc.fc2.weight.data
            new_fc.sigma.data = self.fc.sigma.data
            self.fc = new_fc
            new_out_features = new_dim
            self.n_classes = new_out_features

        else:
            in_features = self.fc.in_features
            out_features = self.fc.out_features

            #print("in_features:", in_features, "out_features:", out_features)
            new_out_features = new_dim
            num_new_classes = new_dim-out_features
            new_fc = SplitCosineLinear(in_features, out_features, num_new_classes)

            new_fc.fc1.weight.data = self.fc.weight.data
            new_fc.sigma.data = self.fc.sigma.data
            self.fc = new_fc
            self.n_classes = new_out_features


    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img, feat=False):
        x = self.conv_layers(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if feat:
            return F.normalize(x, p=2,dim=1)
        else:
            x = self.fc(x)
            return x
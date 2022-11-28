import copy
import math
from bdb import Breakpoint
from builtins import breakpoint

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn
import timm

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.pe[:x.size(0)]
        return self.dropout(x)

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe          = torch.zeros(length, d_model)
    position    = torch.arange(0, length).unsqueeze(1)
    div_term    = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask_all, x_context = None):
        b, n, _, h = *x.shape, self.heads

        if(x_context is None):  x_context = x
        else:                 x_context = x_context

        qkv = (self.to_q(x), *self.to_kv(x_context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # breakpoint()
        masks_np_q   = mask_all[0]
        masks_bert   = mask_all[1]
        masks_np_k   = mask_all[2]
        BS           = masks_np_q.shape[0]
        masks_np_q   = masks_np_q.view(BS, -1)
        masks_np_k   = masks_np_k.view(BS, -1)
        masks_np_    = rearrange(masks_np_q, 'b i -> b () i ()') * rearrange(masks_np_k, 'b j -> b () () j')
        masks_np_    = masks_np_.repeat(1, self.heads, 1, 1)
        # masks_bert   = masks_bert.view(BS, -1)
        # masks_bert_  = rearrange(masks_bert, 'b i -> b () () i')
        # masks_bert_  = masks_bert_.repeat(1, self.heads, masks_bert_.shape[-1], 1)
        dots[masks_np_==0]   = -1e3
        # dots[masks_bert_==1] = -1e3

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim    = dim_head *  heads
        project_out  = not (heads == 1 and dim_head == dim)

        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask_all):
        qkv          = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v      = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots         = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        masks_np     = mask_all[0]
        masks_bert   = mask_all[1]
        
        BS           = masks_np.shape[0]
        
        masks_np     = masks_np.view(BS, -1)
        masks_np_    = rearrange(masks_np, 'b i -> b () i ()') * rearrange(masks_np, 'b j -> b () () j')
        masks_np_    = masks_np_.repeat(1, self.heads, 1, 1)
        
        
        masks_bert   = masks_bert.view(BS, -1)
        masks_bert_  = rearrange(masks_bert, 'b i -> b () () i')
        masks_bert_  = masks_bert_.repeat(1, self.heads, masks_bert_.shape[-1], 1)
        
        
        dots[masks_np_==0]   = -1e3 #-torch.finfo(dots.dtype).max
        dots[masks_bert_==1] = -1e3
        
        del masks_np, masks_np_, masks_bert, masks_bert_
        
        # lower = torch.triu(torch.ones(50, 50)).transpose(0,1)==0
        # lower = lower.unsqueeze(0).unsqueeze(0)
        # lower = lower.repeat(BS, self.heads, 1, 1)
        # dots[lower] = -1e10
        
        attn    = self.attend(dots)
        attn    = self.dropout(attn)

        out     = torch.matmul(attn, v)
        out     = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), 0 #dots

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
    def forward(self, x, mask_np, context = None):
        att_all = []
        for attn, ff, cattn, cff in self.layers:
            x_, att_map = attn(x, mask_all=[mask_np[0], mask_np[1]]) 
            x           = x_ + x
            x           = ff(x) + x
            att_all.append(att_map)

            x_, att_map = cattn(x, mask_all=mask_np, x_context=context) 
            x           = x_ + x
            x           = cff(x) + x
            
        return x, torch.stack(att_all)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask_np):
        att_all = []
        for attn, ff in self.layers:
            x_, _       = attn(x, mask_all=mask_np) 
            x           = x + self.drop_path(x_)
            x           = x + self.drop_path(ff(x))
            # att_all.append(att_map)
            
        return x, 0 #torch.stack(att_all)

class ViT(nn.Module):
    def __init__(self, opt, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., droppath = 0., device=None):
        super().__init__()
        self.opt  = opt
        self.dim  = dim
        self.device = device
        
        delta_                  = 1
        self.delta_             = delta_
        self.class_token        = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pose_token         = nn.Parameter(torch.randn(1, 1, self.dim))
        self.mask_token         = nn.Parameter(torch.randn(self.dim,))
        if(self.opt.masked_mvit):
            self.mask_token_mvit = nn.Parameter(torch.randn(self.opt.extra_feat.mvit.en_dim,))

        if(self.opt.pos_embedding=="learned"):
            if(self.opt.bottle_neck=="conv2i"):
                self.pos_embedding3 = nn.Parameter(torch.randn(1, opt.frame_length+delta_, 1, self.dim))
                
            elif(self.opt.bottle_neck=="conv2j" or self.opt.bottle_neck=="conv_1j"):
                self.pos_embedding5t = nn.Parameter(positionalencoding1d(self.dim, 10000))#.to(self.device)
                self.pos_embedding5s = nn.Parameter(positionalencoding1d(self.dim, 10000))#.to(self.device)
                
                self.register_buffer('pe1', self.pos_embedding5t)
                self.register_buffer('pe2', self.pos_embedding5s)
                
            elif(self.opt.bottle_neck=="conv2k" or self.opt.bottle_neck=="conv2l" or self.opt.bottle_neck=="conv2m" or self.opt.bottle_neck=="conv3l"):
                self.pos_embedding5p = nn.Parameter(positionalencoding2d(self.dim, 250, 10))#.to(self.device)
                self.register_buffer('pe1', self.pos_embedding5p)
            
            else:
                self.pos_embedding  = nn.Parameter(torch.randn(1, (opt.frame_length+delta_)*opt.max_people, self.dim))
                self.pos_embedding2 = nn.Parameter(torch.randn(1, (opt.frame_length+delta_)*opt.max_people, self.dim))
            
        if(self.opt.pos_embedding=="cosine"):
            self.pos_embedding_layer = PositionalEncoding(self.dim, dropout=emb_dropout, max_len=1000)
            # if(self.dim%2==0):
            #     pos_            = positionalencoding1d(self.dim, (opt.frame_length+delta_)*opt.max_people)
            # else:
            #     pos_            = positionalencoding1d(self.dim+1, (opt.frame_length+delta_)*opt.max_people)
            #     pos_            = pos_[:,:-1]
            # pos_                = pos_.unsqueeze(0)
            # self.pos_embedding  = pos_.cuda().float()
            # self.pos_embedding2 = pos_.cuda().float()
            # self.register_buffer('pe1', self.pos_embedding)
            # self.register_buffer('pe2', self.pos_embedding2)

        self.i                  = nn.Identity()
        self.dropout            = nn.Dropout(emb_dropout)
        self.pos_drop           = nn.Dropout(p=0.1)
        
        self.transformer1       = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)
        self.transformer2       = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)

        if("conv" in self.opt.bottle_neck):
            pad                 = self.opt.vit.conv.pad
            stride              = self.opt.vit.conv.stride
            kernel              = stride + 2 * pad
            self.conv_en        = nn.Conv1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)
            self.conv_de        = nn.ConvTranspose1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)
        
        self.pose_shape_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.pose_shape.dim, self.opt.extra_feat.pose_shape.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.pose_shape.mid_dim, self.opt.extra_feat.pose_shape.en_dim),
                                        )
        
        if("hmr" in self.opt.extra_feat.enable):
            self.hmr_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.hmr.dim, self.opt.extra_feat.hmr.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.hmr.mid_dim, self.opt.extra_feat.hmr.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.hmr.mid_dim, self.opt.extra_feat.hmr.en_dim),
                                        )
        if("objects" in self.opt.extra_feat.enable):
            self.objects_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.objects.dim, self.opt.extra_feat.objects.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.objects.mid_dim, self.opt.extra_feat.objects.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.objects.mid_dim, self.opt.extra_feat.objects.en_dim),
                                        )
        if("appe" in self.opt.extra_feat.enable):
            self.appe_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.appe.dim, self.opt.extra_feat.appe.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.appe.mid_dim, self.opt.extra_feat.appe.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.appe.mid_dim, self.opt.extra_feat.appe.en_dim),
                                        )
        if("clip" in self.opt.extra_feat.enable):
            self.clip_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.clip.dim, self.opt.extra_feat.clip.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.clip.mid_dim, self.opt.extra_feat.clip.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.clip.mid_dim, self.opt.extra_feat.clip.en_dim),
                                        )
        if("action" in self.opt.extra_feat.enable):
            self.action_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.action.dim, self.opt.extra_feat.action.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.action.mid_dim, self.opt.extra_feat.action.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.action.mid_dim, self.opt.extra_feat.action.en_dim),
                                        )
        if("mvit" in self.opt.extra_feat.enable):
            self.mvit_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.mvit.dim, self.opt.extra_feat.mvit.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.mvit.mid_dim, self.opt.extra_feat.mvit.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.mvit.mid_dim, self.opt.extra_feat.mvit.en_dim),
                                        )
        
        if("vitpose" in self.opt.extra_feat.enable):
            self.vitpose_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.vitpose.dim, self.opt.extra_feat.vitpose.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.vitpose.mid_dim, self.opt.extra_feat.vitpose.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.vitpose.mid_dim, self.opt.extra_feat.vitpose.en_dim),
                                        )
            
        if("mae" in self.opt.extra_feat.enable):
            self.mae_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.mae.dim, self.opt.extra_feat.mae.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.mae.mid_dim, self.opt.extra_feat.mae.mid_dim),
                                            nn.ReLU(),         
                                            nn.Linear(self.opt.extra_feat.mae.mid_dim, self.opt.extra_feat.mae.en_dim),
                                        )
        
        if(self.opt.use_relative_pose):
            self.relative_pose_encoder       = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.relative_pose.dim, self.opt.extra_feat.relative_pose.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.relative_pose.mid_dim, self.opt.extra_feat.relative_pose.en_dim),
                                        )
            
        if("img" in self.opt.extra_feat.enable):
            # create a resnet50 backbone 
            self.img_encoder = timm.create_model('resnet34', pretrained=True, features_only=True, out_indices=[4])
            self.img_encoder2 = nn.Sequential(
                                            nn.Linear(self.opt.extra_feat.img.dim, self.opt.extra_feat.img.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.opt.extra_feat.img.mid_dim, self.opt.extra_feat.img.en_dim),
                                        )
            
    def bert_mask(self, data, mask_type):
        # print(mask_type)
        if(mask_type=="random"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            indexes        = has_detection.nonzero()
            indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.opt.mask_ratio)]]
            mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
        if(mask_type=="random_x"):
            has_detection  = data['has_detection']>=0
            mask_detection = data['mask_detection']
            for i in range(data['has_detection'].shape[0]):
                indexes        = has_detection[i].nonzero()
                indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.opt.mask_ratio)]]
                mask_detection[i, indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2]] = 1.0
        if(mask_type=="random_y"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            for i in range(data['has_detection'].shape[0]):
                indexes        = has_detection[i].nonzero()
                indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.opt.mask_ratio)]]
                mask_detection[i, indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2]] = 1.0
        if(mask_type=="zero"):
            has_detection  = data['has_detection']==0
            mask_detection = data['mask_detection']
            indexes_mask   = has_detection.nonzero()
            mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
            has_detection = has_detection*0 + 1.0
        if(mask_type=="zero_x"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            # indexes_mask   = has_detection.nonzero()
            # mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
            has_detection = has_detection

        return data, has_detection, mask_detection


    def forward(self, data, mask_type="random"):
        
        data, has_detection, mask_detection = self.bert_mask(data, mask_type)

        if(self.opt.extra_feat.enable=="hmr"):
            pose_  = data['pose_shape'].float()
            hmr_   = data['hmr_emb'].float()
            hmr_en = self.hmr_encoder(hmr_)
            if(self.opt.extra_feat.hmr.en_dim==self.opt.in_feat):
                x       = pose_ + hmr_en
            else:
                x       = torch.cat((pose_, hmr_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="objects"):
            pose_  = data['pose_shape'].float()
            objects_ = data['objects_emb'].float()
            objects_en = self.objects_encoder(objects_)
            if(self.opt.extra_feat.objects.en_dim==self.opt.in_feat):
                x       = pose_ + objects_en
            else:
                x       = torch.cat((pose_, objects_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="appe"):
            pose_  = data['pose_shape'].float()
            appe_ = data['appe_emb'].float()
            appe_en = self.appe_encoder(appe_)
            if(self.opt.extra_feat.appe.en_dim==self.opt.in_feat):
                x       = pose_ + appe_en
            else:
                x       = torch.cat((pose_, appe_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="action"):
            pose_  = data['pose_shape'].float()
            action_ = data['action_emb'].float()
            
            pose_en = self.pose_shape_encoder(pose_)
            action_en = self.action_encoder(action_)
            if(self.opt.extra_feat.action.en_dim==self.opt.in_feat):
                x       = pose_en + action_en
            else:
                x       = torch.cat((pose_en, action_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit"):
            pose_  = data['pose_shape'].float()
            mvit_ = data['mvit_emb'].float()
            
            pose_en = self.pose_shape_encoder(pose_)
            mvit_en = self.mvit_encoder(mvit_)
            if(self.opt.extra_feat.mvit.en_dim==self.opt.in_feat):
                x       = pose_en + mvit_en
            else:
                x       = torch.cat((pose_en, mvit_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="clip"):
            pose_  = data['pose_shape'].float()
            clip_ = data['clip_emb'].float()
            clip_en = self.clip_encoder(clip_)
            if(self.opt.extra_feat.clip.en_dim==self.opt.in_feat):
                x       = pose_ + clip_en
            else:
                x       = torch.cat((pose_, clip_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="vitpose"):
            pose_  = data['pose_shape'].float()
            pose_en = self.pose_shape_encoder(pose_)
            
            vitpose_ = data['vitpose_emb'].float()
            vitpose_en = self.vitpose_encoder(vitpose_)
            
            x = torch.cat((pose_en, vitpose_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
        elif(self.opt.extra_feat.enable=="mae"):
            pose_  = data['pose_shape'].float()
            pose_en = self.pose_shape_encoder(pose_)
            
            mae_ = data['mae_emb'].float()
            mae_en = self.mae_encoder(mae_)
            
            x = torch.cat((pose_en, mae_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
            
        elif(self.opt.extra_feat.enable=="vitpose,img"):
            pose_  = data['pose_shape'].float()
            pose_en = self.pose_shape_encoder(pose_)
            
            vitpose_ = data['vitpose_emb'].float()
            vitpose_en = self.vitpose_encoder(vitpose_)
                    
            a1 = self.img_encoder(data['img'].float())
            a1 = a1[0].mean(dim=-1).mean(dim=-1)
            a2 = self.img_encoder2(a1)
            
            x = torch.cat((pose_en, vitpose_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
            
        
            
        elif(self.opt.extra_feat.enable=="mvit,objects,vitpose"):
            pose_  = data['pose_shape'].float()
            mvit_ = data['mvit_emb'].float()
            objects_ = data['objects_emb'].float()
            vitpose_ = data['vitpose_emb'].float()
            
            pose_en = self.pose_shape_encoder(pose_)
            mvit_en = self.mvit_encoder(mvit_)
            objects_en = self.objects_encoder(objects_)
            vitpose_en = self.vitpose_encoder(vitpose_)
            if(self.opt.extra_feat.mvit.en_dim==self.opt.in_feat):
                x       = pose_en + mvit_en + vitpose_en
            else:
                x       = torch.cat((pose_en, mvit_en, objects_en, vitpose_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
        elif(self.opt.extra_feat.enable=="mvit,hmr,objects,clip,vitpose"):
            pose_    = data['pose_shape'].float()
            
            mvit_    = data['mvit_emb'].float()
            hmr_     = data['hmr_emb'].float()
            clip_    = data['clip_emb'].float()
            objects_ = data['objects_emb'].float()
            vitpose_ = data['vitpose_emb'].float()
            
            mvit_en    = self.mvit_encoder(mvit_)
            hmr_en     = self.hmr_encoder(hmr_)
            clip_en    = self.clip_encoder(clip_)
            objects_en = self.objects_encoder(objects_)
            vitpose_en = self.vitpose_encoder(vitpose_)
            
            x = torch.cat((pose_, mvit_en, hmr_en, clip_en, objects_en, vitpose_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit,hmr"):
            pose_    = data['pose_shape'].float()
            
            mvit_    = data['mvit_emb'].float()
            hmr_     = data['hmr_emb'].float()
            
            pose_en    = self.pose_shape_encoder(pose_)
            mvit_en    = self.mvit_encoder(mvit_)
            hmr_en     = self.hmr_encoder(hmr_)
            
            x = torch.cat((pose_en, mvit_en, hmr_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit,clip"):
            pose_    = data['pose_shape'].float()
            
            mvit_    = data['mvit_emb'].float()
            clip_    = data['clip_emb'].float()
            
            pose_en    = self.pose_shape_encoder(pose_)
            mvit_en    = self.mvit_encoder(mvit_)
            clip_en    = self.clip_encoder(clip_)
            
            x = torch.cat((pose_en, mvit_en, clip_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit,vitpose"):
            pose_    = data['pose_shape'].float()
            
            mvit_    = data['mvit_emb'].float()
            vitpose_ = data['vitpose_emb'].float()
            
            pose_en    = self.pose_shape_encoder(pose_)
            mvit_en    = self.mvit_encoder(mvit_)
            vitpose_en = self.vitpose_encoder(vitpose_)
            
            if(self.opt.masked_mvit and not("zero" in mask_type)):
                # at 0.5% of the time, we mask the mvit
                p1 = torch.rand(mvit_en.shape[0])
                loc_p1 = p1>0.5
                mvit_en[loc_p1, :, :, :] = self.mask_token_mvit
            
            x = torch.cat((pose_en, mvit_en, vitpose_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit,objects"):
            pose_    = data['pose_shape'].float()
            
            mvit_    = data['mvit_emb'].float()
            objects_ = data['objects_emb'].float()
            
            pose_en    = self.pose_shape_encoder(pose_)
            mvit_en    = self.mvit_encoder(mvit_)
            objects_en = self.objects_encoder(objects_)
            
            x = torch.cat((pose_en, mvit_en, objects_en), dim=-1)
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="mvit_only"):
            
            mvit_    = data['mvit_emb'].float()
            mvit_en    = self.mvit_encoder(mvit_)
            
            x = mvit_en
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
        elif(self.opt.extra_feat.enable=="vitpose_only"):
            
            vitpose_    = data['vitpose_emb'].float()
            vitpose_en    = self.vitpose_encoder(vitpose_)
            
            x = vitpose_en
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
            
        else:
            pose_   = data['pose_shape'].float()
            pose_en = self.pose_shape_encoder(pose_)
            x       = pose_en
            x[mask_detection[:, :, :, 0]==1] = self.mask_token
            
        if(self.opt.use_relative_pose):
            rel_pose = data['relative_pose'].float()
            rel_pose = self.relative_pose_encoder(rel_pose)
            x = x + rel_pose

        BS, T, P, dim    = x.size()
        x                = x.view(BS, T*P, dim)
        loss             = torch.zeros(1).to(x.device)
        
        if(self.opt.bottle_neck=="conv"):
            x                = x + self.pos_embedding[:, self.opt.max_people:] 
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])

            x = x.transpose(1, 2)
            x = self.conv_en(x)
            x = self.conv_de(x)
            x = x.transpose(1, 2)
            x = x.contiguous()

            x                = x + self.pos_embedding2[:, self.opt.max_people:]
            has_detection    = has_detection*0 + 1
            mask_detection   = mask_detection*0
            x, attentions    = self.transformer2(x, [has_detection, mask_detection])
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            
        if(self.opt.bottle_neck=="conv_1"): 
            x                = x + self.pos_embedding[:, self.opt.max_people:] 
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])

            x_ = x.view(BS, T, P, dim)
            x_ = x_.permute(0, 2, 3, 1)
            x_ = x_.reshape(BS*P, dim, T)
            x_ = self.conv_en(x_)
            x_ = self.conv_de(x_)
            x_ = x_.reshape(BS, P, dim, T)
            x_ = x_.permute(0, 3, 1, 2)
            x_ = x_.reshape(BS, T*P, dim)
            x_ = x_.contiguous()

            x                = x + self.pos_embedding2[:, self.opt.max_people:]
            has_detection    = has_detection*0 + 1
            mask_detection   = mask_detection*0
            x, attentions    = self.transformer2(x, [has_detection, mask_detection])
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            
        if(self.opt.bottle_neck=="conv2"):
            # import pdb; pdb.set_trace()
            x_               = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, self.opt.max_people, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, self.opt.max_people, 1).to(mask_detection.device), mask_detection], dim=1)           
            
            x                = x_ + self.pos_embedding
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
        if(self.opt.bottle_neck=="conv2i"):
            # import pdb; pdb.set_trace()
            x_               = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, self.opt.max_people, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, self.opt.max_people, 1).to(mask_detection.device), mask_detection], dim=1)           
            
            x                = x_ + self.pos_embedding3.repeat(1, 1, self.opt.max_people, 1).view(1, (self.opt.frame_length+self.delta_)*self.opt.max_people, dim)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
        if(self.opt.bottle_neck=="conv2j"):

            x                = x + self.pos_embedding5t[None, :self.opt.frame_length, None, :].repeat(1, 1, self.opt.max_people, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            x                = x + self.pos_embedding5s[None, None, :self.opt.max_people, :].repeat(1, self.opt.frame_length, 1, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            
        if(self.opt.bottle_neck=="conv2k"):
            # av = self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people]
            # ab = av.reshape(1, dim, self.opt.frame_length*self.opt.max_people)
            # ac = ab.permute(0, 2, 1)
            
            x                = x + self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people].reshape(1, dim, self.opt.frame_length*self.opt.max_people).permute(0, 2, 1)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            
        
        
        if(self.opt.bottle_neck=="conv2l"):
            # av = self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people]
            # ab = av.reshape(1, dim, self.opt.frame_length*self.opt.max_people)
            # ac = ab.permute(0, 2, 1)
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, self.opt.max_people, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, self.opt.max_people, 1).to(mask_detection.device), mask_detection], dim=1)       
            
            x                = x + self.pos_embedding5p[None, :, :self.opt.frame_length+1, :self.opt.max_people].reshape(1, dim, (self.opt.frame_length+1)*self.opt.max_people).permute(0, 2, 1)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
        if(self.opt.bottle_neck=="conv3l"):
            # av = self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people]
            # ab = av.reshape(1, dim, self.opt.frame_length*self.opt.max_people)
            # ac = ab.permute(0, 2, 1)
            x = torch.concat([a2.unsqueeze(1).repeat(1, self.opt.max_people, 1) , x], dim=1)
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 2, self.opt.max_people, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 2, self.opt.max_people, 1).to(mask_detection.device), mask_detection], dim=1)       
            
            x                = x + self.pos_embedding5p[None, :, :self.opt.frame_length+2, :self.opt.max_people].reshape(1, dim, (self.opt.frame_length+2)*self.opt.max_people).permute(0, 2, 1)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
            cls_token       = x[:, :self.opt.max_people, :]
            pose_tokens     = x[:, 2*self.opt.max_people:, :]
            x = torch.concat([cls_token, pose_tokens], dim=1)
            
        if(self.opt.bottle_neck=="conv2m"):
            
            x                = x + self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people].reshape(1, dim, (self.opt.frame_length)*self.opt.max_people).permute(0, 2, 1)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
            x_ = x.view(BS, T, P, dim)
            x_ = x_.permute(0, 2, 3, 1)
            x_ = x_.reshape(BS*P, dim, T)
            x_ = self.conv_en(x_)
            x_ = self.conv_de(x_)
            x_ = x_.reshape(BS, P, dim, T)
            x_ = x_.permute(0, 3, 1, 2)
            x_ = x_.reshape(BS, T*P, dim)
            x  = x_.contiguous()

            has_detection    = has_detection*0 + 1
            mask_detection   = mask_detection*0
            x                = x + self.pos_embedding5p[None, :, :self.opt.frame_length, :self.opt.max_people].reshape(1, dim, (self.opt.frame_length)*self.opt.max_people).permute(0, 2, 1)
            x, attentions    = self.transformer2(x, [has_detection, mask_detection])
            
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            
            
        if(self.opt.bottle_neck=="conv_1j"):
            
            x                = x + self.pos_embedding5t[None, :self.opt.frame_length, None, :].repeat(1, 1, self.opt.max_people, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            x                = x + self.pos_embedding5s[None, None, :self.opt.max_people, :].repeat(1, self.opt.frame_length, 1, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
            # x_ = x.view(BS, T, P, dim)
            # x_ = x_.permute(0, 2, 3, 1)
            # x_ = x_.reshape(BS*P, dim, T)
            # x_ = self.conv_en(x_)
            # x_ = self.conv_de(x_)
            # x_ = x_.reshape(BS, P, dim, T)
            # x_ = x_.permute(0, 3, 1, 2)
            # x_ = x_.reshape(BS, T*P, dim)
            # x_ = x_.contiguous()
            
            x = x.transpose(1, 2)
            x = self.conv_en(x)
            x = self.conv_de(x)
            x = x.transpose(1, 2)
            x = x.contiguous()
            
            x                = x + self.pos_embedding5t[None, :self.opt.frame_length, None, :].repeat(1, 1, self.opt.max_people, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            x                = x + self.pos_embedding5s[None, None, :self.opt.max_people, :].repeat(1, self.opt.frame_length, 1, 1).view(1, self.opt.frame_length*self.opt.max_people, dim)#.to(x.device)
            
            has_detection    = has_detection*0 + 1
            mask_detection   = mask_detection*0
            x, attentions    = self.transformer2(x, [has_detection, mask_detection])
            
            x = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)     
            
        if(self.opt.bottle_neck=="conv2p"):
            # import pdb; pdb.set_trace()
            x_               = torch.concat([self.class_token.repeat(BS, self.opt.max_people, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, self.opt.max_people, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, self.opt.max_people, 1).to(mask_detection.device), mask_detection], dim=1)           
            
            x                = self.pos_embedding_layer(x)
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
        if(self.opt.bottle_neck=="conv2x"):
            x_               = torch.concat([self.class_token.repeat(BS, 1, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, 1, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, 1, 1).to(mask_detection.device), mask_detection], dim=1)           
            
            x                = x_ + self.pos_embedding
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            x = x_ + x
            
        if(self.opt.bottle_neck=="conv2y"):
            x                = torch.concat([self.class_token.repeat(BS, 1, 1), x], dim=1)
            
        if(self.opt.bottle_neck=="conv3"):
            x                = torch.concat([self.class_token.repeat(BS, 1, 1), x], dim=1)
            has_detection    = torch.concat([torch.ones(BS, 1, 1, 1).to(has_detection.device), has_detection], dim=1)
            mask_detection   = torch.concat([torch.zeros(BS, 1, 1, 1).to(mask_detection.device), mask_detection], dim=1)           
            
            x                = x + self.pos_embedding
            x, attentions    = self.transformer1(x, [has_detection, mask_detection])
            
            x_cls  = x[:, :1, :]
            x      = x[:, 1:, :]
            x = x.transpose(1, 2)
            x = self.conv_en(x)
            x = self.conv_de(x)
            x = x.transpose(1, 2)
            x = x.contiguous()
            x = torch.concat([x_cls, x], dim=1)

            x                = x + self.pos_embedding2
            has_detection    = has_detection*0 + 1
            mask_detection   = mask_detection*0
            x, attentions    = self.transformer2(x, [has_detection, mask_detection])            

        return x, loss

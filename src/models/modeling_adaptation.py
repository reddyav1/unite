import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from .modeling_finetune import Block, DropPath, Mlp, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange, repeat


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 


class AdaptationVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_frames=16, tubelet_size=2,
                 use_checkpoint=False, checkpoint_num=0, use_learnable_pos_emb=False, clip_return_layers=[6,7,8,9,10,11],
                 clip_student_return_interval=1, use_cls_token=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            num_frames=num_frames, tubelet_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        self.return_index = clip_return_layers
        # for i in clip_return_layers:
        #     self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        # self.return_index = [6] # TODO: REMOVE THIS LATER
        print(f'Student return index: {self.return_index}')

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
            num_patches += 1
        
        self.use_learnable_pos_emb = use_learnable_pos_emb
        if use_learnable_pos_emb:
            print('Use learnable position embedding')
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, clip_only=False):
        x = self.patch_embed(x)
        B, _, C = x.shape

        # Add the CLS token if requested
        if hasattr(self, 'cls_token'):
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply positional embeddings
        if self.use_learnable_pos_emb:
            x = x + self.pos_embed.type_as(x).to(x.device)
        else:
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        # Apply mask
        if hasattr(self, 'cls_token'): # if cls token is used, remove it before applying mask
            cls_tokens = x[:, 0:1, :]
            x = x[:, 1:, :]
            x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
            x_vis = torch.cat([cls_tokens, x_vis], dim=1) # add cls token back
        else:
            x_vis = x[~mask].reshape(B, -1, C)
        x_clip_vis = []

        for idx, blk in enumerate(self.blocks):
            # pass input through transformer block
            if self.use_checkpoint and idx < self.checkpoint_num:
                x_vis = checkpoint.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)
            # collect this layer's representations if it is in return_index
            if idx in self.return_index:
                x_clip_vis.append(x_vis)
            if idx == max(self.return_index) and clip_only:
                break

        x_clip_vis = self.norm(torch.stack(x_clip_vis))
        return x_vis, x_clip_vis

    def forward(self, x, mask, clip_only=False):
        x, x_clip_vis = self.forward_features(x, mask, clip_only=clip_only)
        x_clip_vis = self.head(x_clip_vis)

        if clip_only:
            return x_clip_vis
        else:
            x = self.head(self.norm(x))
            return x, x_clip_vis


class Linear_Decoder(nn.Module):
    """ Linear Decoder for CLIP feature alignment """
    def __init__(self, num_classes=768, embed_dim=768, 
                 norm_layer=nn.LayerNorm, clip_norm_type='l2'):
        super().__init__()
        self.clip_norm_type = clip_norm_type

        self.head = nn.Linear(embed_dim, num_classes)
        self.norm =  norm_layer(num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.clip_norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x


class AdaptationVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_cls_token=False,
                 use_checkpoint=False,
                 checkpoint_num=0,
                 num_frames=16,
                 tubelet_size=2,
                 # clip,
                 clip_decoder_embed_dim=768,
                 clip_output_dim=512,
                 clip_norm_type='l2',
                 clip_return_layers=[6,7,8,9,10,11],
                 clip_student_return_interval=1,
                ):
        super().__init__()

        self.encoder = AdaptationVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer,
            init_values=init_values,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
            use_learnable_pos_emb=use_learnable_pos_emb,
            clip_return_layers=clip_return_layers,
            clip_student_return_interval=clip_student_return_interval,
            use_cls_token=use_cls_token,
        )

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                num_classes=clip_output_dim, 
                embed_dim=clip_decoder_embed_dim, 
                norm_layer=norm_layer, 
                clip_norm_type=clip_norm_type
            ) for _ in range(len(clip_return_layers))
        ])

        self.clip_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, clip_decoder_embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token', 'clip_mask_token', 'clip_pos_embed'}

    def forward(self, x, mask, clip_only=False):
        if clip_only:
            x_clip_vis = self.encoder(x, mask, clip_only=True)
        else:
            x_vis, x_clip_vis = self.encoder(x, mask, clip_only=False) # [K, B, N_vis, C_e]
        if hasattr(self.encoder, 'cls_token'):
            cls_token = x_clip_vis[-1, :, 0, :]
            raw_encoder_outputs = x_clip_vis # we will return encoder outputs with cls token
            x_clip_vis = x_clip_vis[:, :, 1:, :] # remove cls token for CLIP alignment
        else:
            raw_encoder_outputs = x_clip_vis
        
        # align CLIP
        K, B, _, C_CLIP = x_clip_vis.shape
        expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
        clip_pos_emd_vis = expand_clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        x_clip_full = x_clip_vis + clip_pos_emd_vis # [K, B, N_vis, C_d_clip]

        x_clip = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip.append(clip_decoder(x_clip_full[idx]))
        x_clip = torch.stack(x_clip) # align and normalize

        # x_vis : [B, N_vis, C_o] -- outputs of last encoder layer
        # x_clip_vis : [K, B, N_vis, C_o] -- encoder outputs at K alignment layers
        # x_clip : [K, B, N_vis, C_d_clip] -- CLIP decoder outputs at K alignment layers
        
        if clip_only:
            return x_clip
        else:
            return x_vis, x_clip
    

@register_model
def adaptation_umt_base_patch16_224(pretrained=False, **kwargs):
    model = AdaptationVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 

@register_model
def adaptation_umt_large_patch16_224(pretrained=False, **kwargs):
    model = AdaptationVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = adaptation_umt_base_patch16_224()

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, 16, 224, 224))
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)
    mask = torch.cat([
        torch.ones(1, 8 * int(14 * 14 * 0.75)),
        torch.zeros(1, 8 * int(14 * 14 * 0.25)),
    ], dim=-1).to(torch.bool)
    print(model(torch.rand(1, 3, 16, 224, 224), mask)[1].shape)

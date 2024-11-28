import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from model.build_sam import build_sam_vit_b
from model.clip import build_model
from einops import rearrange
from functools import partial
from .layers import CrossModalFPNDecoder, QueryDecoder, FusionModule
from model.utils import PositionEmbeddingSine2D
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from torch.nn import init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
act_func = nn.GELU()
class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        r: int
    ):
        super().__init__()
        self.r = r
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)
        self.act = act_func
        self.w_identity = torch.eye(self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        B,H,W,_ = qkv.shape

        q = qkv[..., :self.dim].reshape(B,H*W,-1).clone()
        v = qkv[..., self.dim*2:].reshape(B,H*W,-1).clone()

        new_q = self.linear_b_q(self.act(self.linear_a_q(q)))
        new_v = self.linear_b_v(self.act(self.linear_a_v(v)))

        new_query = qkv[...,:self.dim] + new_q.reshape(B,H,W,-1)
        new_value = qkv[..., 2*self.dim:] + new_v.reshape(B,H,W,-1)
        qkv[...,:self.dim] = new_query
        qkv[..., 2*self.dim:] =  new_value
        return qkv

class UniHRSOD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len=12
        self.kernel=3
        clip_model = torch.jit.load(cfg.Model.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.Model.word_len).float()
        self.backbone_visual = build_sam_vit_b(checkpoint=cfg.Model.sam_pretrain)

        for t_layer_i, blk in enumerate(self.backbone_visual.blocks):
            if t_layer_i < self.len:
                self.backbone_visual.blocks[t_layer_i].attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, 16*4)
            else:
                self.backbone_visual.blocks[t_layer_i] = nn.Identity()

        # Cross-modal Fusion
        self.fusion = FusionModule(d_model=cfg.Model.fusion_dim)
        self.fusion_proj = nn.Linear(512, cfg.Model.fusion_dim)
        self.visual_fusion_pos = PositionEmbeddingSine2D(cfg.Model.fusion_dim//2, normalize=True)   # half channel of vision feature

        self.all_fusion_proj = nn.ModuleList()
        self.all_fusion_module = nn.ModuleList()
        self.all_fusion_pos = nn.ModuleList()

        for i in range(3):
            self.all_fusion_proj.append(nn.Linear(512, 768))
            self.all_fusion_module.append(FusionModule(d_model=768)),
            self.all_fusion_pos.append(PositionEmbeddingSine2D(384, normalize=True))

        # Pixel Decoder
        self.pixel_decoder = CrossModalFPNDecoder(cfg.Model.pixel_decoder_in, cfg.Model.pixel_decoder_conv, cfg.Model.pixel_decoder_mask, norm=None)

        # Query Decoder
        self.query_decoder = QueryDecoder(d_model=cfg.Model.d_model, num_enc=cfg.Model.num_enc, num_dec=cfg.Model.num_dec, in_visual_dim=cfg.Model.visual_in, in_text_dim=cfg.Model.text_in,
                                        return_intermediate_dec=cfg.Model.aux_loss)


        embed_dim = 768
        conv_inplane = 64
        drop_rate = 0.
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        deform_num_heads = 6
        init_values = 1e-6
        interaction_indexes = [[0, 1], [2, 3], [4, 9], [10, 11]]
        with_cffn = True,
        cffn_ratio = 0.25
        deform_ratio = 1
        use_extra_extractor = True
        with_cp = True
        n_points = 4
        self.drop_path_rate = 0.1
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.interaction_indexes = interaction_indexes
        self.norm = nn.LayerNorm(768)
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.add_vit_feature = True
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    def forward(self, img, word):
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool() # B, L

        # vis: 4✕ / 8✕ / 16✕ / 32✕
        # word: b, length, 512
        # state: b, 1024


        deform_inputs1, deform_inputs2 = deform_inputs(img)
        # SPM forward
        c1, c2, c3, c4 = self.spm(img)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.backbone_visual.patch_embed(img)

        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.norm(x)

        # bs, n, dim = x.shape
        B_, N_, C_ = x.shape
        res = int(np.sqrt(N_))

        x = x.view(B_, res, res, -1)
        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)

        x = self.pos_drop(x + self.backbone_visual.pos_embed)

        # Interaction
        outs = list()

        x = x.view(B_, -1, C_)
        bs, n, dim = x.shape

        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]

            x, c = layer(x, c, self.backbone_visual.blocks[indexes[0]:indexes[-1] + 1],
                            deform_inputs1, deform_inputs2, H, W)

            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        x1 = self.norm1(c1)
        x2 = self.norm2(c2)
        x3 = self.norm3(c3)
        x4 = self.norm4(c4)
        vis = [x1,x2,x3,x4]

        word, state = self.backbone.encode_text(word)

        b, c, h, w = vis[-1].shape
        visual_fusion_pos = self.visual_fusion_pos(vis[-1])
        visual_fusion_pos = rearrange(visual_fusion_pos, 'b c h w -> (h w) b c', b=b)
        v2l_input = rearrange(vis[-1], 'b c h w -> (h w) b c', b=b)
        word_proj = self.fusion_proj(word)
        word_proj = rearrange(word_proj, 'b l c -> l b c')
        v2l_input = self.fusion(tgt=v2l_input,
                                memory=word_proj,
                                memory_key_padding_mask=pad_mask,
                                pos=None,
                                query_pos=visual_fusion_pos
        )
        for i in range(len(vis)-1):
            _, _, hi, wi = vis[i].shape
            visual_fusion_pos = self.all_fusion_pos[i](vis[i])
            visual_fusion_pos = rearrange(visual_fusion_pos, 'b c h w -> (h w) b c', b=b)
            word_proj = self.all_fusion_proj[i](word)
            word_proj = rearrange(word_proj, 'b l c -> l b c')
            fusion_input = rearrange(vis[i], 'b c h w -> (h w) b c', b=b)

            fusion_input = self.all_fusion_module[i](tgt=fusion_input,
                                memory=word_proj,
                                memory_key_padding_mask=pad_mask,
                                pos=None,
                                query_pos=visual_fusion_pos) 
            vis[i] = rearrange(fusion_input, '(h w) b c -> b c h w', h=hi, w=wi)

        v2l_input = rearrange(v2l_input, '(h w) b c -> b c h w', h=h, w=w)

        # L to V
        visual_input = []

        for i in range(len(vis)):
            visual_input.append(vis[i])
        visual_input[-1] = v2l_input
        visual_output = self.pixel_decoder(visual_input, word, pad_mask)

        # V to L
        language_output = self.query_decoder(v2l_input, state)

        _, _, h, w = visual_output.shape
        final_output = []
        if not self.cfg.Model.aux_loss:
            pred = torch.bmm(language_output, visual_output.flatten(2))
            pred = rearrange(pred, 'b l (h w) -> b l h w', h=h, w=w)   
        else:
            for l, q in enumerate(language_output):
                

                pred = torch.bmm(language_output[l], visual_output.flatten(2))
                pred = rearrange(pred, 'b l (h w) -> b l h w', h=h, w=w)

                final_output.append(pred)

        return final_output

if __name__ == '__main__':
    model = UniHRSOD()

    batch_size = 2
    channel = 3
    height = 512
    width = 512
    input_data = torch.randn(batch_size, channel, height, width)

    # 将输入数据传递给模型进行前向传播
    output = model(input_data)

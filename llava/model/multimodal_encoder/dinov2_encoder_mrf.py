import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config, AutoImageProcessor

class Dinov2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.patch_size = 14  # Dinov2-B/14
        # self.target_sizes = [224, 336]  # 多尺度
        # self.target_sizes = [224, 336, 518]  # 多尺度
        self.target_sizes = [224, 378, 518]  # 多尺度
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return
        # 这里的 processor 只用于归一化（不让它自动 resize/crop）
        img_size = max(self.target_sizes)
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, size={"height": img_size, "width": img_size}, crop_size={"height": img_size, "width": img_size})
        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        x = image_forward_outs.hidden_states[self.select_layer]  # (B, 1+N, C)
        if self.select_feature == 'patch':
            x = x[:, 1:]  # 去掉 CLS
        elif self.select_feature == 'cls_patch':
            pass
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return x  # (B, N, C)

    def _run_one_scale(self, images_chw, side: int):
        """
        images_chw: (B, 3, H, W) tensor in [0,1] or [0,255]
        side: 224 / 336 / 518
        returns: (B, N=grid^2, C) patch tokens at this scale (未插值到最大网格)
        """
        # 1) 先把输入 resize 到 side×side（不改变通道/类型/设备）

        pixel_values = F.interpolate(images_chw, size=(side, side), mode='bilinear', align_corners=False)

        # 3) 过 Dinov2，取指定层的 patch token
        out = self.vision_tower(pixel_values, output_hidden_states=True)
        feats = self.feature_select(out).to(images_chw.dtype)  # (B, N, C)

        return feats  # (B, (side/14)^2, C)

    def _tokens_to_grid(self, tokens, side):
        """
        tokens: (B, N, C), side: 图片边长
        return: (B, C, H, W) with H=W=side//patch
        """
        B, N, C = tokens.shape
        g = side // self.patch_size
        assert g * g == N, f"N={N}不等于({g}^2)；side={side}, patch={self.patch_size}"
        grid = tokens.view(B, g, g, C).permute(0, 3, 1, 2).contiguous()  # (B, C, g, g)
        return grid

    def _grid_to_tokens(self, grid):
        """
        grid: (B, C, H, W)  -> (B, H*W, C)
        """
        B, C, H, W = grid.shape
        tokens = grid.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        return tokens

    @torch.no_grad()
    def multi_scale_forward(self, images):
        """
        images: Tensor (B,3,518,518) 或 (B,3,H,W)，数值范围 [0,1] / [0,255] 都可
        流程：224/336/518 -> token -> 小网格插值到最大网格(56x56) -> 最终在通道维拼接
        返回: (B, 56*56, 3*C)
        """
        # 提取三个尺度的 patch tokens
        feats_list = []
        grids_list = []
        for s in self.target_sizes:
            t = self._run_one_scale(images, s)                 # (B, (s/14)^2, C)
            g = self._tokens_to_grid(t, s)                     # (B, C, s/14, s/14)
            grids_list.append(g)

        # 最大网格 784/14=56
        max_g = self.target_sizes[-1] // self.patch_size  # 56

        # 把小网格插值到 (56,56)
        up_grids = []
        for g, side in zip(grids_list, self.target_sizes):
            cur_g = side // self.patch_size
            if cur_g != max_g:
                g_up = F.interpolate(g, size=(max_g, max_g), mode='bilinear', align_corners=False)  # (B,C,56,56)
            else:
                g_up = g
            up_grids.append(g_up)

        # 在通道维拼接 -> (B, C*3, 56, 56) -> 再摊平成 (B, 56*56, C*3)
        up_cat = torch.cat(up_grids, dim=1)                    # (B, 3C, 56, 56)
        out = self._grid_to_tokens(up_cat)                     # (B, 3136, 3C)
        return out

    @torch.no_grad()
    def forward(self, images):
        """
        如果你想替换原先的单尺度 forward，就让它直接走 multi_scale_forward。
        images: (B,3,H,W)
        """
        return self.multi_scale_forward(images)

    @property
    def dummy_feature(self):
        # 注意 dummy 的通道维需要 x3
        return torch.zeros(1, (max(self.target_sizes)//self.patch_size)**2, len(self.target_sizes)*self.hidden_size, device=self.device, dtype=self.dtype)
        # return torch.zeros(1, (336//self.patch_size)**2, len(self.target_sizes)*self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return len(self.target_sizes) * self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return 518 // self.patch_size

    @property
    def num_patches(self):
        g = self.num_patches_per_side
        return g * g  # 3136

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoImageProcessor, Dinov2Model


class HybridVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.clip_vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.dino_vision_tower_name = "facebook/dinov2-base"

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.clip_vision_tower_name)
        self.dino_image_processor = AutoImageProcessor.from_pretrained(self.dino_vision_tower_name,
                                                              size={"height": 336, "width": 336}, crop_size={"height": 336, "width": 336})

        self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.clip_vision_tower_name, device_map=device_map)
        self.clip_vision_tower.requires_grad_(False)

        self.dino_vision_tower = Dinov2Model.from_pretrained(self.dino_vision_tower_name, device_map=device_map)
        self.dino_vision_tower.requires_grad_(False)

        self.is_loaded = True

    def clip_feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _forward_dino_single_scale(self, images):
        image_forward_outs = self.dino_vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        return image_forward_outs.hidden_states[-1][:, 1:].to(images.dtype)

    def _forward_dino_multiscale(self, images):
        assert images.shape[-2:] == (336, 336), "Input images must be of the same size as 336"

        low_res_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        high_res_features = self._forward_dino_single_scale(images)
        low_res_features = self._forward_dino_single_scale(low_res_images)

        low_res_features_grid = low_res_features.view(low_res_features.shape[0], 16, 16, low_res_features.shape[-1]).permute(0, 3, 1, 2)  # [B, D, G, G]
        low_res_features_up = F.interpolate(
            low_res_features_grid,
            size=(24, 24),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).contiguous().view(low_res_features.shape[0], -1, low_res_features.shape[-1])  #

        # [B, N, D] + [B, N, D] -> [B, N, 2D]
        image_features = torch.cat([high_res_features, low_res_features_up], dim=-1)
        return image_features

    @torch.no_grad()
    def forward(self, clip_images, dino_images):
        if type(clip_images) is list:
            image_features = []
            for clip_image, dino_image in zip(clip_images, dino_images):
                clip_image_forward_out = self.clip_vision_tower(clip_image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                clip_image_feature = self.clip_feature_select(clip_image_forward_out).to(clip_image.dtype)
                dino_image_feature = self._forward_dino_multiscale(dino_image.unsqueeze(0))
                image_feature = torch.concatenate([clip_image_feature, dino_image_feature], dim=-1)
                image_features.append(image_feature)
        else:
            clip_image_forward_outs = self.clip_vision_tower(clip_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            clip_image_features = self.clip_feature_select(clip_image_forward_outs).to(clip_images.dtype)
            dino_image_feature = self._forward_dino_multiscale(dino_images)
            image_features = torch.concatenate([clip_image_features, dino_image_feature], dim=-1)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1024 + 768 * 2, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.dino_vision_tower.dtype

    @property
    def device(self):
        return self.dino_vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1024 + 768 * 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



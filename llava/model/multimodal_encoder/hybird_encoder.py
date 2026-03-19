import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoImageProcessor, Dinov2Model


class HybirdVisionTower(nn.Module):
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

    @torch.no_grad()
    def forward(self, clip_images, dino_images):
        if type(clip_images) is list:
            image_features = []
            for clip_image, dino_image in zip(clip_images, dino_images):
                clip_image_forward_out = self.clip_vision_tower(clip_image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                clip_image_feature = self.clip_feature_select(clip_image_forward_out).to(clip_image.dtype)
                dino_image_feature = self.dino_vision_tower(dino_image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True).hidden_states[-1].to(dino_image.dtype)
                dino_image_feature = dino_image_feature[:, 1:] # remove cls token
                image_feature = torch.concatenate([clip_image_feature, dino_image_feature], dim=-1)
                image_features.append(image_feature)
        else:
            clip_image_forward_outs = self.clip_vision_tower(clip_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            clip_image_features = self.clip_feature_select(clip_image_forward_outs).to(clip_images.dtype)
            dino_image_feature = self.dino_vision_tower(dino_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True).hidden_states[-1].to(dino_images.dtype)
            dino_image_feature = dino_image_feature[:, 1:] # remove cls token
            image_features = torch.concatenate([clip_image_features, dino_image_feature], dim=-1)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1024 + 768, device=self.device, dtype=self.dtype)

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
        return 1024 + 768

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



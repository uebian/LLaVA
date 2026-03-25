import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import Siglip2VisionConfig, Siglip2VisionModel, Siglip2ImageProcessor

class _Siglip2ImageProcessor:
    def __init__(self, processor: Siglip2ImageProcessor, image_size: int = 384, patch_size: int = 16):
        self._processor = processor
        self._image_size = int(image_size)
        self._patch_size = int(patch_size)
        self._max_num_patches = (self._image_size // self._patch_size) ** 2

    def _maybe_resize(self, images):
        try:
            from PIL import Image
            pil_available = True
        except Exception:
            pil_available = False

        def resize_one(img):
            if pil_available and isinstance(img, Image.Image):
                return img.resize((self._image_size, self._image_size), resample=Image.BICUBIC)
            return img

        if isinstance(images, list):
            return [resize_one(x) for x in images]
        return resize_one(images)

    def preprocess(self, images, *args, **kwargs):
        images = self._maybe_resize(images)
        kwargs.setdefault("patch_size", self._patch_size)
        kwargs.setdefault("max_num_patches", self._max_num_patches)
        results = self._processor.preprocess(images, *args, **kwargs)
        assert results['pixel_values'].ndim == 3
        return results

    def __call__(self, images, *args, **kwargs):
        images = self._maybe_resize(images)
        kwargs.setdefault("patch_size", self._patch_size)
        kwargs.setdefault("max_num_patches", self._max_num_patches)
        return self._processor(images, *args, **kwargs)

    @property
    def image_mean(self):
        return self._processor.image_mean

    @property
    def crop_size(self):
        return {"height": self._image_size, "width": self._image_size}

    @property
    def size(self):
        return {"height": self._image_size, "width": self._image_size}

    def __getattr__(self, name):
        return getattr(self._processor, name)


class SigLip2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # For SigLip2, we set select_layer to -2
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.image_size = 384
        self.patch_size = 16

        base_processor = Siglip2ImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor_384 = _Siglip2ImageProcessor(
            base_processor,
            image_size=384,
            patch_size=self.patch_size,
        )
        self.image_processor_256 = _Siglip2ImageProcessor(
            base_processor,
            image_size=256,
            patch_size=self.patch_size,
        )
        self.image_processor = (self.image_processor_384, self.image_processor_256)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Siglip2VisionConfig.from_pretrained(self.vision_tower_name)
            # LLaVA expects these for patch bookkeeping.
            self.cfg_only.image_size = self.image_size
            self.cfg_only.patch_size = self.patch_size

    def load_model(self, device_map=None):  # load model and image processor
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_tower = Siglip2VisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # LLaVA expects these for patch bookkeeping.
        self.vision_tower.config.image_size = self.image_size
        self.vision_tower.config.patch_size = self.patch_size
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs): # Selects features from the model output based on configuration.
        # SigLIP2 has no CLIP-style CLS token in the patch sequence.
        hidden_states = image_forward_outs.hidden_states
        image_features = hidden_states[self.select_layer]
        if self.select_feature in ('patch', 'cls_patch'):
            return image_features
        raise ValueError(f'Unexpected select feature: {self.select_feature}')

    def _build_nnaflex_inputs(self, pixel_values: torch.Tensor, *, expected_side: Optional[int] = None):
        # pixel_values is expected to be (B, N, C) tokens produced by Siglip2ImageProcessor.
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 3:
            raise ValueError(f"SigLIP2 expects tokenized pixel_values of shape (B, N, C), got {tuple(pixel_values.shape)}")

        batch_size, num_tokens, _ = pixel_values.shape
        pixel_attention_mask = torch.ones((batch_size, num_tokens), device=pixel_values.device, dtype=torch.int32)

        side = int(round(num_tokens ** 0.5))
        if side * side != num_tokens:
            raise ValueError(f"SigLIP2 expects a square number of tokens, got {num_tokens}")
        if expected_side is not None:
            assert side == expected_side
        spatial_shapes = torch.tensor([[side, side]] * batch_size, device=pixel_values.device, dtype=torch.long)
        return pixel_values, pixel_attention_mask, spatial_shapes

    def _forward_single_scale(self, pixel_values: torch.Tensor, *, expected_side: int):
        pixel_values, pixel_attention_mask, spatial_shapes = self._build_nnaflex_inputs(
            pixel_values,
            expected_side=expected_side,
        )
        image_forward_outs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_hidden_states=True,
        )
        return self.feature_select(image_forward_outs)

    def _upsample_square_tokens(self, token_features: torch.Tensor, *, from_side: int, to_side: int):
        # token_features: [B, from_side*from_side, D] -> [B, to_side*to_side, D]
        batch_size, num_tokens, dim = token_features.shape
        expected_tokens = from_side * from_side
        if num_tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens for a {from_side}x{from_side} grid, got {num_tokens}")

        grid = token_features.view(batch_size, from_side, from_side, dim).permute(0, 3, 1, 2)  # [B, D, H, W]
        grid_up = F.interpolate(
            grid,
            size=(to_side, to_side),
            mode='bilinear',
            align_corners=False,
        )
        return grid_up.permute(0, 2, 3, 1).contiguous().view(batch_size, to_side * to_side, dim)

    @torch.no_grad()
    def forward(self, clip_images, dino_images):
        """Fuse SigLIP2 features from two token streams.

        - clip_images: tokenized 384x384 stream (24x24 = 576 tokens)
        - dino_images: tokenized 256x256 stream (16x16 = 256 tokens), upsampled to 24x24 then concatenated

        Returns: [B, 576, 2*D] (or list of such tensors)
        """
        high_side = 24
        low_side = 16

        if type(clip_images) is list:
            image_features = []
            for clip_image, low_image in zip(clip_images, dino_images):
                high_tokens = clip_image.to(device=self.device, dtype=self.dtype)
                low_tokens = low_image.to(device=self.device, dtype=self.dtype)

                high_features = self._forward_single_scale(high_tokens, expected_side=high_side).to(clip_image.dtype)
                low_features = self._forward_single_scale(low_tokens, expected_side=low_side).to(low_image.dtype)


                low_features_up = self._upsample_square_tokens(low_features, from_side=low_side, to_side=high_side)
                fused = torch.cat([high_features, low_features_up.to(high_features.dtype)], dim=-1)
                image_features.append(fused)
        else:
            high_tokens = clip_images.to(device=self.device, dtype=self.dtype)
            low_tokens = dino_images.to(device=self.device, dtype=self.dtype)

            high_features = self._forward_single_scale(high_tokens, expected_side=high_side).to(clip_images.dtype)
            low_features = self._forward_single_scale(low_tokens, expected_side=low_side).to(dino_images.dtype)
            low_features_up = self._upsample_square_tokens(low_features, from_side=low_side, to_side=high_side)
            image_features = torch.cat([high_features, low_features_up.to(high_features.dtype)], dim=-1)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size * 2, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # feature dim is doubled after concatenation (384 stream + upsampled 256 stream)
        return self.config.hidden_size * 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
# from .clip_encoder_mrf import CLIPVisionTower, CLIPVisionTowerS2
# from .dinov2_encoder import Dinov2VisionTower, Dinov2VisionTowerS2
from .dinov2_encoder_mrf import Dinov2VisionTower
# from .hybrid_encoder import HybridVisionTower
from .hybrid_encoder_mrf import HybridVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)


    if vision_tower.startswith("facebook/dinov2"):
        # if use_s2:
        #     return Dinov2VisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        # else:
        return Dinov2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower.startswith("hybrid"):
        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

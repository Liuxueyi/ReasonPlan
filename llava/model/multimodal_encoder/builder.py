import os
import torch
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .map_encoder import MTREncoder, MTREncoderConfig
from mmcv.models import build_model

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if 'clip' in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        elif "siglip" in vision_tower:
            return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_map_encoder(**kwargs):
    map_encoder_cfg = MTREncoderConfig(**kwargs)
    model = MTREncoder(map_encoder_cfg)
    return model.to(torch.bfloat16)


def build_perception_encoder(perception_config):
    model = build_model(
        perception_config.model,
        train_cfg=perception_config.get('train_cfg'),
        test_cfg=perception_config.get('test_cfg'))
    
    # NOTE: The following code is used to initialize the backbone with `resnet50-19c8e357.pth`. 
    # NOTE: It will raise the bug `Attempted to set the storage of a tensor on device "cuda:0" to a storage on different device "cpu"` in `legacy_load()`.
    # NOTE: Maybe it is because the ckpt is too old? We choose to load a whole pretrained perception model out of this function instead.
    # from deepspeed import zero
    # with zero.GatheredParameters(model.parameters()):
    #     model.init_weights()

    return model.to(torch.bfloat16)


  
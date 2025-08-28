import torch
import torch.nn as nn
import math


class SpatialPool(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.mode = model_args.mm_spatial_pool_mode
        self.stride = model_args.mm_spatial_pool_stride
        self.out_channels = getattr(model_args, "mm_spatial_pool_out_channels", vision_tower.hidden_size)

        if self.mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "conv":
            self.pool = nn.Conv2d(in_channels=vision_tower.hidden_size, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}.")

    def forward(self, image_features, images, *args, **kwargs):
        ori_W = int(math.sqrt(image_features.shape[1] * images.shape[3] // images.shape[2]))
        ori_H = int(ori_W * images.shape[2] // images.shape[3])

        B, _, F = image_features.shape

        image_features_spatial = image_features.view(B, ori_H, ori_H, F).permute(0, 3, 1, 2) # FIXME ori_H, ori_H
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    @property
    def config(self):
        return {
            "mm_resampler_type": "spatial_pool",
            "mm_spatial_pool_stride": self.stride,
            "mm_spatial_pool_mode": self.mode,
            "mm_spatial_pool_out_channels": self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels

class BEVSpatialPool(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.mode = model_args.perception_spatial_pool_mode
        self.stride = model_args.perception_spatial_pool_stride
        self.out_channels = getattr(model_args, "perception_encoder_out_feature", 256)
        
        self.ori_H = model_args.perception_config['bev_h_']
        self.ori_W = model_args.perception_config['bev_w_']

        if self.mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.mode == "conv":
            self.pool = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.stride, stride=self.stride)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}.")

    def forward(self, bev_features, *args, **kwargs):
        B, _, F = bev_features.shape

        bev_features_spatial = bev_features.view(B, self.ori_H, self.ori_W, F).permute(0, 3, 1, 2)
        bev_features_spatial_pool = self.pool(bev_features_spatial)

        return bev_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()

    @property
    def config(self):
        return {
            "perception_resampler_type": "spatial_pool",
            "perception_spatial_pool_stride": self.stride,
            "perception_spatial_pool_mode": self.mode,
            "perception_encoder_out_feature": self.out_channels,
        }

    @property
    def hidden_size(self):
        return self.out_channels


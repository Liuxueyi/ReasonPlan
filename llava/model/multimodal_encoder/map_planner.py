import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .map_encoder import MTREncoder, MTREncoderConfig
from .ego_encoder import EgoInfoEncoder, EgoInfoEncoderConfig
from .ego_map_attention import EgoMapAttention, EgoMapAttentionConfig

class PointLanePlannerConfig(PretrainedConfig):
    model_type = "point_lane_planner"

    def __init__(self, in_channels=15, out_channels=256, num_heads=8, num_layers=2, hidden_dim=256, use_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_pe = use_pe

class PointLanePlanner(PreTrainedModel):
    config_class = PointLanePlannerConfig
    _no_split_modules = ["PointLanePlanner"]

    def __init__(self, config: PointLanePlannerConfig):
        super().__init__(config)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.use_pe = config.use_pe
        self.transformer_block_num = config.num_layers

        mtr_encoder_config = MTREncoderConfig(in_channels=self.in_channels, hidden_dim=self.hidden_dim, out_channels=self.out_channels, num_layers=self.transformer_block_num, use_pe=self.use_pe)
        ego_info_encoder = EgoInfoEncoderConfig(feature_size=self.out_channels)
        ego_map_attention_config = EgoMapAttentionConfig(embed_size=self.out_channels, num_heads=self.num_heads, num_decoder_layers=self.transformer_block_num)
        self.map_encoder = MTREncoder(mtr_encoder_config)
        self.ego_encoder = EgoInfoEncoder(ego_info_encoder)
        self.ego_map_attention = EgoMapAttention(ego_map_attention_config)

    def forward(self, current_info, road_pts, lane_mask, point_mask):
        B, S, P, M = road_pts.shape
        map_info, map_pos = self.map_encoder(road_pts, point_mask)
        ego_info = self.ego_encoder(current_info)
        ego_map_attention = self.ego_map_attention(ego_info, map_info, lane_mask, map_pos)

        return ego_map_attention
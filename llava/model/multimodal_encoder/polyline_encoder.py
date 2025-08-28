import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertConfig
from transformers.modeling_outputs import BaseModelOutput

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            self.num_batches_tracked += 1
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.weight * x_normalized + self.bias

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

class PositionalEncoding(nn.Module):
    def __init__(self, num_positions, position_dim):
        super().__init__()
        self.position_dim = position_dim
        self.position_encodings = self._generate_position_encodings(num_positions, position_dim)

    # def _generate_position_encodings(self, num_positions, position_dim):
    #     position_indices = torch.arange(num_positions).unsqueeze(1).float()
    #     div_term = torch.exp(torch.arange(0, position_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / position_dim))
    #     position_encodings = torch.zeros(num_positions, position_dim)
    #     position_encodings[:, 0::2] = torch.sin(position_indices * div_term)
    #     position_encodings[:, 1::2] = torch.cos(position_indices * div_term)
    #     return position_encodings
    
    def _generate_position_encodings(self, num_positions, position_dim, num_freq_bands=8):
        position_indices = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, num_freq_bands).float() * -(torch.log(torch.tensor(10000.0)) / num_freq_bands))
        freqs = position_indices * div_term * 2 * torch.pi
        pos_encodings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) 
        if position_dim % 2 != 0:
            pos_encodings = torch.cat([pos_encodings, torch.zeros(num_positions, 1)], dim=-1)
        return pos_encodings[:, :position_dim]
    
    def forward(self, batch_size, num_polylines):
        position_encodings = self.position_encodings.unsqueeze(0).unsqueeze(0)
        position_encodings = position_encodings.repeat(batch_size, num_polylines, 1, 1)
        return position_encodings
    
class PointNetPolylineConfig(BertConfig):
    def __init__(self, 
                 in_channels=3, 
                 hidden_dim=64, 
                 num_layers=3, 
                 num_pre_layers=1, 
                 out_channels=None, 
                 use_position_encoding=False, 
                 position_dim=16, 
                 num_positions=5,  # be the same with number of points in map input
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_pre_layers = num_pre_layers
        self.out_channels = out_channels
        self.use_position_encoding = use_position_encoding
        self.position_dim = position_dim
        self.num_positions = num_positions

class PointNetPolylineEncoder(PreTrainedModel):
    config_class = PointNetPolylineConfig
    _no_split_modules = ["PointNetPolylineEncoder"]
    def __init__(self, config):
        super().__init__(config)
        self.use_position_encoding = config.use_position_encoding
        if self.use_position_encoding:
            self.position_encoding = PositionalEncoding(config.num_positions, config.position_dim)
            self.in_channels = config.in_channels + config.position_dim
        else:
            self.in_channels = config.in_channels

        self.pre_mlps = build_mlps(
            c_in=self.in_channels,
            mlp_channels=[config.hidden_dim] * config.num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=config.hidden_dim * 2,
            mlp_channels=[config.hidden_dim] * (config.num_layers - config.num_pre_layers),
            ret_before_act=False
        )
        
        if config.out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=config.hidden_dim, mlp_channels=[config.hidden_dim, config.out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        batch_size, num_polylines, num_points_each_polylines, _ = polylines.shape

        if self.use_position_encoding:
            position_encodings = self.position_encoding(batch_size, num_polylines)
            polylines = torch.cat([polylines, position_encodings[:, :, :num_points_each_polylines, :].to(polylines.device)], dim=-1).type(torch.bfloat16)

        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        feature_buffers = feature_buffers.max(dim=2)[0]
        
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
            
        return BaseModelOutput(last_hidden_state=feature_buffers)
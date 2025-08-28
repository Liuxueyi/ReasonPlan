from transformers import PreTrainedModel, BertConfig
from .transformerblock import TransformerBlockConfig, TransformerBlock
from .polyline_encoder import PointNetPolylineConfig, PointNetPolylineEncoder
import torch, math
import torch.nn as nn
from torch.nn import functional as F

class MTREncoderConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = kwargs['in_channels']
        self.hidden_dim = kwargs['hidden_dim']
        self.out_channels = kwargs['out_channels']
        self.num_layers = kwargs['num_layers']
        self.use_pe = kwargs['use_pe']

class MTREncoder(PreTrainedModel):
    config_class = MTREncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=5,
            num_pre_layers=3,
            out_channels=config.out_channels,
            use_position_encoding=True,
            position_dim=16,
        )

        # # check if polyline is initialized !
        # for name, param in self.map_polyline_encoder.named_parameters():
        #     print(f"Layer: {name} | Shape: {param.shape}")

        self_attn_layers = []
        for _ in range(config.num_layers):
            transformer_config = TransformerBlockConfig(
                d_model=config.out_channels,
                nhead=8,
                dropout=0.1,
                dim_feedforward=config.out_channels * 4,
            )
            self_attn_layers.append(TransformerBlock(transformer_config))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = config.out_channels
        
        self.point_head = nn.Linear(config.out_channels, 5*2)
        
        self._no_split_modules = ["map_polyline_encoder", "self.self_attn_layers"]
        self.config = config
    
    def build_polyline_encoder(self, 
        in_channels, hidden_dim, num_layers, 
        num_pre_layers=1, out_channels=None, 
        use_position_encoding=False, position_dim=16
        ):
        config = PointNetPolylineConfig(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels,
            use_position_encoding=use_position_encoding,
            position_dim=position_dim
        )
        ret_polyline_encoder = PointNetPolylineEncoder(config)
        return ret_polyline_encoder

    def apply_global_attn(self, x, x_mask, x_pos):
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embedding = self.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)
        pos_embedding = pos_embedding.type(torch.bfloat16)
        for layer in self.self_attn_layers:
            x_t = layer(
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            ).last_hidden_state
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out, pos_embedding
    
    def gen_sineembed_for_position(self, pos_tensor, hidden_dim=256):
        half_hidden_dim = hidden_dim // 2
        scale = 2 * math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=torch.bfloat16, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos
    
    def compute_map_polylines_center(self, road_pts, point_mask, eps=1e-8):
        batch_size, num_polylines, num_points_each_polylines, _ = road_pts.shape
        xy_coords = road_pts[:, :, :, :2]  
        point_mask_expanded = point_mask.unsqueeze(-1).expand_as(xy_coords)
        masked_xy_coords = xy_coords * point_mask_expanded.float()
        valid_points_count = point_mask.sum(dim=2, keepdim=True).float()
        
        valid_points_count_with_eps = valid_points_count.clone()
        valid_points_count_with_eps[valid_points_count_with_eps == 0] = eps
        
        sum_valid_points = masked_xy_coords.sum(dim=2)
        center_points_xy = sum_valid_points / valid_points_count_with_eps
        
        center_points_xy[valid_points_count.squeeze(-1) == 0] = 0
        
        center_points = torch.zeros((batch_size, num_polylines, 2), dtype=road_pts.dtype)
        
        center_points[:, :, :2] = center_points_xy
        
        map_polylines_center = center_points.unsqueeze(1)
        
        return map_polylines_center
    
    def forward(self, road_pts, same_lane, point_mask):
        num_polylines = road_pts.shape[1]

        map_polylines_center = self.compute_map_polylines_center(road_pts, point_mask).to(device=road_pts.device)
        
        map_polylines_feature = self.map_polyline_encoder(road_pts, point_mask).last_hidden_state
        map_valid_mask = (point_mask.sum(dim=-1) > 0)  
        global_token_feature = map_polylines_feature 
        global_token_mask = map_valid_mask
        global_token_pos = map_polylines_center.squeeze(1)

        global_token_feature, pos_embedding = self.apply_global_attn(
            x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
        )

        map_polylines_feature = global_token_feature
        assert map_polylines_feature.shape[1] == num_polylines

        pred_map = self.point_head(map_polylines_feature).reshape(-1, num_polylines, 5, 2)
        map_aux_losses = self.cal_aux_lossed(pred_map, road_pts, point_mask)

        # select map info 
        expanded_same_lane = same_lane.unsqueeze(-1).expand(-1, -1, -1, self.config.out_channels)
        expanded_map_info = map_polylines_feature.unsqueeze(1).expand(-1, same_lane.shape[1], -1, -1)
        selected_map_info = expanded_map_info * expanded_same_lane
        count = same_lane.sum(dim=-1, keepdim=True).to(dtype=selected_map_info.dtype)

        count_with_no_zero = count.clone()
        count_with_no_zero[count_with_no_zero == 0] = 1
        selected_map_info = selected_map_info.sum(dim=-2) / count_with_no_zero

        return selected_map_info, map_aux_losses

    def cal_aux_lossed(self, pred_map, road_pts, point_mask):
        """
        Compute the map loss.

        Args:
        pred_map (torch.Tensor): Predicted map with shape (B, S, P, 2).
        road_pts (torch.Tensor): Ground truth road points with shape (B, S, P, 10).
        point_mask (torch.Tensor): Mask indicating valid lanes with shape (B, S, P).

        Returns:
        torch.Tensor: The computed loss.
        """
        losses = {}
        
        # Extract the relevant dimensions from road_pts
        road_pts_relevant = road_pts[..., :2]  # Shape: (B, S, P, 2)

        # Apply lane mask
        lane_mask_expanded = point_mask.unsqueeze(-1)  # Shape: (B, S, P, 1)
        masked_pred_map = pred_map * lane_mask_expanded  # Shape: (B, S, P, 2)
        masked_road_pts = road_pts_relevant * lane_mask_expanded  # Shape: (B, S, P, 2)

        # Calculate the loss
        loss = F.mse_loss(masked_pred_map, masked_road_pts, reduction='sum')

        # Normalize the loss by the number of valid points
        num_valid_points = point_mask.sum().float()
        loss = loss / num_valid_points

        losses['map_loss'] = loss
        return losses
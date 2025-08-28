#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import transformers
import deepspeed
from deepspeed import zero

from .multimodal_encoder.builder import build_vision_tower, build_map_encoder, build_perception_encoder
from .multimodal_resampler.builder import build_vision_resampler, build_map_resampler, build_perception_resampler
from .multimodal_projector.builder import build_vision_projector, build_map_projector, build_perception_projector, build_obj_projector

from llava.constants import *

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print

from torch.nn import functional as F

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        hidden_size = config.hidden_size
        self.ego_vel_mlp = nn.Sequential(
                                nn.Linear(2, hidden_size), # 32 4
                                nn.SiLU(),
                                nn.Linear(hidden_size, hidden_size),
                            ) 
        self.ego_acc_mlp = nn.Sequential(
                        nn.Linear(2, hidden_size), # 32 4
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size),
                    ) 
        self.command_mlp = nn.Sequential(
                                nn.Linear(6, hidden_size),
                                nn.SiLU(),
                                nn.Linear(hidden_size, hidden_size),
                            ) 
        
        # self.waypoint_query = nn.Embedding(6, hidden_size)

        # self.planning_head = nn.Sequential(
        #                     nn.Linear(hidden_size, hidden_size),
        #                     nn.SiLU(),
        #                     nn.Linear(hidden_size, 2) # 16*3 6*2
        #                 )

        # Initialize clip vision modules for loading pretrained weights
        if hasattr(config, "mm_vision_tower"):
             # TODO map_encoder is always loaded from path config.mm_vision_tower, instead of llava ckpt.
             # To unfreeze the map_encoder, during evaluation, the map_encoder should be loaded from the fine-tuned llava ckpt. 
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

            rank0_print("build vision tower done")

        # Initialize map modules for loading pretrained weights
        if hasattr(config, "use_map_encoder") and config.use_map_encoder:
            self.map_encoder = build_map_encoder(
                                    in_channels=self.config.map_encoder_in_feature, 
                                    hidden_dim=self.config.map_encoder_hidden_feature, 
                                    out_channels=self.config.map_encoder_out_feature,
                                    num_layers=self.config.map_encoder_num_layers,
                                    use_pe=self.config.map_use_pe,
                                )
            self.map_resampler = build_map_resampler()
            self.map_projector = build_map_projector(config)

            rank0_print("build map encoder done")

        # Initialize perception modules for loading pretrained weights
        if hasattr(config, "use_perception_encoder") and config.use_perception_encoder:
            from mmcv import Config
            perception_config_mmcv = Config(config.perception_config)
            self.perception_encoder = build_perception_encoder(perception_config_mmcv)
            self.perception_resampler = build_perception_resampler(config)
            self.perception_projector = build_perception_projector(config)
            self.obj_projector = build_obj_projector(config)

            rank0_print("build perception encoder done")

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_map_encoder(self):
        map_encoder = getattr(self, "map_encoder", None)
        if type(map_encoder) is list:
            map_encoder = map_encoder[0]
        return map_encoder

    def get_perception_encoder(self):
        perception_encoder = getattr(self, "perception_encoder", None)
        if type(perception_encoder) is list:
            perception_encoder = perception_encoder[0]
        return perception_encoder

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    def initialize_map_modules(self, zero_stage, model_args):
        map_encoder_in_feature = model_args.map_encoder_in_feature
        map_encoder_hidden_feature = model_args.map_encoder_hidden_feature
        map_encoder_out_feature = model_args.map_encoder_out_feature
        map_encoder_num_layers = model_args.map_encoder_num_layers
        map_use_pe = model_args.map_use_pe

        self.config.map_encoder_pretrained_path = model_args.map_encoder_pretrained_path
        self.config.map_encoder_in_feature = map_encoder_in_feature
        self.config.map_encoder_hidden_feature = map_encoder_hidden_feature
        self.config.map_encoder_out_feature = map_encoder_out_feature
        self.config.map_encoder_num_layers = map_encoder_num_layers
        self.config.map_use_pe = map_use_pe

        self.config.map_projector_type = getattr(model_args, "map_projector_type", "linear")

        # build map encoder
        if self.get_map_encoder() == None:
            with deepspeed.zero.Init(enabled=(zero_stage==3), config_dict_or_path=transformers.integrations.deepspeed_config()):
                self.map_encoder = build_map_encoder(
                                    in_channels=map_encoder_in_feature, 
                                    hidden_dim=map_encoder_hidden_feature, 
                                    out_channels=map_encoder_out_feature, 
                                    num_layers=map_encoder_num_layers, 
                                    use_pe=map_use_pe
                                )
                self.map_resampler = build_map_resampler()
                self.map_projector = build_map_projector(self.config)
            
            for p in self.map_resampler.parameters():
                p.requires_grad = True
            for p in self.map_projector.parameters():
                p.requires_grad = True

            # Load pretrained weights
            if model_args.map_encoder_pretrained_path is not None:
                map_encoder_weights = torch.load(model_args.map_encoder_pretrained_path, map_location="cpu")
                with zero.GatheredParameters(self.map_encoder.parameters(), modifier_rank=0):
                    try:
                        self.map_encoder.load_state_dict(map_encoder_weights)
                    except Exception as e:
                        rank0_print(f"Failed to load map encoder weights from {model_args.map_encoder_pretrained_path}")
                        self.map_encoder.load_state_dict(map_encoder_weights, strict=True)

                rank0_print(f"Loaded map encoder weights from {model_args.map_encoder_pretrained_path}")

    def initialize_perception_modules(self, zero_stage, model_args):
        self.config.perception_config = model_args.perception_config

        perception_resampler_type = model_args.perception_resampler_type
        perception_spatial_pool_mode = model_args.perception_spatial_pool_mode
        perception_spatial_pool_stride = model_args.perception_spatial_pool_stride
        perception_projector_type = model_args.perception_projector_type
        perception_encoder_out_feature = model_args.perception_config._dim_
        
        obj_projector_type = model_args.obj_projector_type
        obj_encoder_out_feature = model_args.perception_config._dim_

        self.config.perception_encoder_pretrained_path = model_args.perception_encoder_pretrained_path
        self.config.perception_resampler_type = perception_resampler_type
        self.config.perception_spatial_pool_mode = perception_spatial_pool_mode
        self.config.perception_spatial_pool_stride = perception_spatial_pool_stride
        self.config.perception_projector_type = perception_projector_type
        self.config.perception_encoder_out_feature = perception_encoder_out_feature
        self.config.obj_projector_type = obj_projector_type
        self.config.obj_encoder_out_feature = obj_encoder_out_feature
        
        # build perception encoder
        if self.get_perception_encoder() == None:
            with deepspeed.zero.Init(enabled=(zero_stage==3), config_dict_or_path=transformers.integrations.deepspeed_config()):
                self.perception_encoder = build_perception_encoder(model_args.perception_config)
                self.perception_resampler = build_perception_resampler(self.config)
                self.perception_projector = build_perception_projector(self.config)
                self.obj_projector = build_obj_projector(self.config)
            
            for p in self.perception_resampler.parameters():
                p.requires_grad = True
            for p in self.perception_projector.parameters():
                p.requires_grad = True
            for p in self.obj_projector.parameters():
                p.requires_grad = True
            
            # load pretrained weights
            if model_args.perception_encoder_pretrained_path is not None:
                perception_encoder_weights = torch.load(model_args.perception_encoder_pretrained_path, map_location="cpu")
                with zero.GatheredParameters(self.perception_encoder.parameters(), modifier_rank=0):
                    self.perception_encoder.load_state_dict(perception_encoder_weights['state_dict'], strict=False)
                rank0_print(f"Loaded perception encoder weights from {model_args.perception_encoder_pretrained_path}")

    def del_vision_modules(self):
        if hasattr(self, "vision_tower"):
            delattr(self, "vision_tower")
        if hasattr(self, "vision_resampler"):
            delattr(self, "vision_resampler")
        if hasattr(self, "mm_projector"):
            delattr(self, "mm_projector")
        if hasattr(self, "image_newline"):
            delattr(self, "image_newline")

    def del_map_modules(self):
        if hasattr(self, "map_encoder"):
            delattr(self, "map_encoder")
        if hasattr(self, "map_resampler"):
            delattr(self, "map_resampler")
        if hasattr(self, "map_projector"):
            delattr(self, "map_projector")

    def del_perception_modules(self):
        if hasattr(self, "perception_encoder"):
            delattr(self, "perception_encoder")
        if hasattr(self, "perception_resampler"):
            delattr(self, "perception_resampler")
        if hasattr(self, "perception_projector"):
            delattr(self, "perception_projector")
        if hasattr(self, "obj_projector"):
            delattr(self, "obj_projector")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_map_encoder(self):
        return self.get_model().get_map_encoder()
    
    def get_perception_encoder(self):
        return self.get_model().get_perception_encoder()

    def encode_images(self, images):         # images: (batsh_size * 5, 336, 336)
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_map(self, lanes, same_lane, points_mask):
        map_features, map_aux_losses = self.get_model().get_map_encoder()(lanes, same_lane, points_mask)
        map_features = self.get_model().map_resampler(map_features)
        map_features = self.get_model().map_projector(map_features)
        
        if hasattr(self.config, "map_loss_weight") and self.config.map_loss_weight is not None:
            map_aux_losses = {k: v * self.config.map_loss_weight for k, v in map_aux_losses.items()}

        return map_features, map_aux_losses

    def encode_perception(self, map_features, **kwargs):
        # perception_features = self.get_model().get_perception_encoder().extract_bev_feat_only(**kwargs)
        perception_features, obj_features, perception_aux_losses = self.get_model().get_perception_encoder().extract_bev_obj_feat(map_features, **kwargs)
        perception_features = self.get_model().perception_resampler(perception_features)
        perception_features = self.get_model().perception_projector(perception_features)
        
        obj_features = self.get_model().obj_projector(obj_features)

        if self.get_model().training:
            if hasattr(self.config, "perception_loss_weight") and self.config.perception_loss_weight is not None:
                perception_aux_losses = {k: v * self.config.perception_loss_weight for k, v in perception_aux_losses.items()}

        return perception_features, obj_features, perception_aux_losses

    # def encode_ego_state(self, current_state):
    #     ego_state_features = self.get_model().ego_state_mlp(current_state)
    #     return ego_state_features

    def prepare_clip_img_features(self, images, image_sizes):
        assert images is not None and image_sizes is not None
    
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] # [10, 3, 384, 384]
            concat_images = torch.cat([image for image in images], dim=0) # torch.Size([10, 3, 384, 384]) torch.Size([20, 3, 384, 384])
            image_features = self.encode_images(concat_images) # torch.Size([10, 729, 1024])
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0) # [10, 729, 1024]  ([10, 729, 1024], [10, 729, 1024])
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                # image_features = [x.flatten(0, 1) for x in image_features] # image_features[0].shape torch.Size([7290, 1024])  ([7290, 1024], [7290, 1024])
                final_features = []
                for image_f in image_features:
                    # front_image_f = image_f[:1].flatten(0, 1) # [729*5, 1024]
                    front_image_f = image_f[:5].flatten(0, 1) # [729*5, 1024]
                    # final_tensors = [front_image_f] + [image_f[5]] + [image_f[6]] + [image_f[7]] + [image_f[8]] + [image_f[9]] + [image_f[10]] + [image_f[11]] + [image_f[12]] + [image_f[13]]
                    final_tensors = [front_image_f] + [image_f[5]] + [image_f[6]] + [image_f[7]] + [image_f[8]] + [image_f[9]]
                    # final_tensors = [front_image_f]
                    final_features.append(final_tensors)
                    
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise ValueError(f"Unexpected image_aspect_ratio: {image_aspect_ratio}")
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # return image_features
        return final_features

    def prepare_map_features(self, lanes, same_lane, points_mask):
        assert lanes is not None and points_mask is not None
        map_features, map_aux_losses = self.encode_map(lanes, same_lane, points_mask)
        return map_features, map_aux_losses

    def prepare_perception_features(self, map_features, **kwargs):
        perception_features, obj_features, perception_aux_losses = self.encode_perception(map_features, **kwargs)
        return perception_features, obj_features, perception_aux_losses

    def prepare_ego_state_features(self, current_state):
        assert current_state is not None
        ego_state_features = self.encode_ego_state(current_state)
        return ego_state_features
    
    def prepare_command_features(self, command):
        assert command is not None
        command_features = self.get_model().command_mlp(command)
        return command_features
    
    def prepare_navi_features(self, navigation):
        assert navigation is not None
        navi_features = self.get_model().navi_mlp(navigation)
        return navi_features

    def encode_ego_state(self, velocity, accel, cmd):
        # ego_state_features = torch.cat((velocity, accel), dim=1)
        ego_vel_features = self.get_model().ego_vel_mlp(velocity).unsqueeze(1)
        ego_acc_features = self.get_model().ego_acc_mlp(accel).unsqueeze(1)
        cmd_features = self.get_model().command_mlp(cmd).unsqueeze(1)
        ego_state_features = torch.cat((ego_vel_features, ego_acc_features, cmd_features), dim=1)

        return ego_state_features

    def concatenate_tokens(self, cur_input_ids, cur_labels, image_features, ego_state_features):
        """
        Concatenate the input tokens with the multimodal features. Similar to the `tokenizer_token` function in `llava/mm_utils.py`.
        1. Get the indexes of multimodal tokens.
        2. Split the input tokens into text tokens and multimodal tokens.
        3. Encode the text tokens with text tokenizer.
        4. Insert the multimodal features into the input embeddings by the token indexes.
        """
        
        # STEP 1: Get the indexes of multimodal tokens
        image_indexes = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()

        indexes = image_indexes 
        indexes.sort()

        # STEP 2: Split the input tokens into text tokens and multimodal tokens
        cur_input_ids_only_text = []
        cur_labels_only_text = []
        prev = 0
        for index in indexes:
            cur_input_ids_only_text.append(cur_input_ids[prev: index])
            cur_labels_only_text.append(cur_labels[prev: index])
            prev = index + 1
        cur_input_ids_only_text.append(cur_input_ids[prev:])
        cur_labels_only_text.append(cur_labels[prev:])
        
        # STEP 3: Encode the text tokens with text tokenizer
        split_sizes = [x.shape[0] for x in cur_labels_only_text] # [37, 777]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_only_text)) # [814, 1024]
        cur_input_embeds_only_text = torch.split(cur_input_embeds, split_sizes, dim=0)

        # STEP 4: Insert the multimodal features into the input embeddings by the token indexes
        cur_new_input_embeds = []
        cur_new_labels = []
        cur_attention_mask = []
        cur_ego_state_tag = []
        img_idx = 0
        for index in indexes:
            cur_new_input_embeds.append(cur_input_embeds_only_text[indexes.index(index)]) # [37, 1024]
            cur_new_labels.append(cur_labels_only_text[indexes.index(index)])
            cur_attention_mask.append(torch.ones(cur_input_embeds_only_text[indexes.index(index)].shape[0], device=cur_labels.device, dtype=torch.bool))
            cur_ego_state_tag.append(torch.zeros(cur_input_embeds_only_text[indexes.index(index)].shape[0], device=cur_labels.device, dtype=torch.bool))
            
            if index in image_indexes:
                # print("img_idx:", img_idx)
                # print('image_indexes:', image_indexes)
                # print("image_features:", len(image_features))
                cur_feature = image_features[img_idx] # torch.Size([7290, 1024]) # TODO only support one image. [image_indexes.index(index)] [3645, 1024]
                if img_idx == 0 and ego_state_features is not None:
                    cur_feature = torch.cat((ego_state_features, cur_feature), dim=0)
                    # print('cat ego status')
                cur_feature_mask = torch.ones((cur_feature.shape[0],), device=cur_labels.device, dtype=torch.bool)
                cur_ego_state_tag_temp = torch.zeros((cur_feature.shape[0],), device=cur_labels.device, dtype=torch.bool)
                img_idx += 1
            else:
                raise ValueError(f"Unexpected multimodal token index: {index}")

            cur_new_input_embeds.append(cur_feature)
            cur_new_labels.append(torch.full((cur_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_attention_mask.append(cur_feature_mask)
            cur_ego_state_tag.append(cur_ego_state_tag_temp)

        cur_new_input_embeds.append(cur_input_embeds_only_text[-1])
        cur_new_labels.append(cur_labels_only_text[-1])
        cur_attention_mask.append(torch.ones((cur_input_embeds_only_text[-1].shape[0],), device=cur_labels.device, dtype=torch.bool))
        cur_ego_state_tag.append(torch.zeros((cur_input_embeds_only_text[-1].shape[0],), device=cur_labels.device, dtype=torch.bool))

        return cur_new_input_embeds, cur_new_labels, cur_attention_mask, cur_ego_state_tag, image_indexes

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None, \
            current_state=None, command=None, navigation=None, navi_mask=None, lanes=None, lanes_mask=None, points_mask=None, same_lane=None, **kwargs,
        ):
        """
        Prepare inputs and labels for multimodal model.
        1. Prepare multimodal features.
        2. Concatenate tokens with multimodal features.
        """
        
        if self.config.use_clip_img_encoder and images is None \
            or input_ids.shape[1] == 1: # NOTE for efficient decoding
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        # STEP 1: Prepare multimodal features
        aux_losses = {}

        if self.config.use_clip_img_encoder:
            image_features = self.prepare_clip_img_features(images, image_sizes) # images.shape torch.Size([1, 10, 3, 384, 384]) torch.Size([2, 10, 3, 384, 384])
        else:
            image_features = [None] * input_ids.shape[0]
        if 'velocity' in kwargs.keys():
            velocity = kwargs['velocity']
            accel = kwargs['accel']
            cmd = kwargs['cmd']
            ego_state_features = self.encode_ego_state(velocity, accel, cmd) # [bs, 2, 1024]
            # image_features[0] = torch.cat((image_features[0], ego_state_features), dim=1)
        else:
            ego_state_features = [None]*input_ids.shape[0]
        if 'images_pred' in kwargs and self.model.config.nsp_enable:
            pre_image = kwargs['images_pred']
            if self.config.use_clip_img_encoder:
                with torch.no_grad():
                    image_preds = self.prepare_clip_img_features(pre_image, image_sizes) # images.shape torch.Size([1, 10, 3, 384, 384]) torch.Size([2, 10, 3, 384, 384])
            image_pred_features = []
            for i in range(len(image_preds)):
                image_pred = torch.cat(image_preds[i], dim=0) # [2, 10, 3, 384, 384]
                image_pred_features.append(image_pred)
            image_pred_features = torch.stack(image_pred_features, dim=0) # [2, 10, 3, 384, 384]
            if self.model.config.only_front:
                aux_losses['image_pred_features'] = image_pred_features[:,:729,:]
            else:
                aux_losses['image_pred_features'] = image_pred_features[:,:5*729,:]


        # STEP 2: Concatenate tokens with multimodal features

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_attention_mask = []
        ego_state_tag = []
        image_indexes = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_new_input_embeds, cur_new_labels, cur_attention_mask, cur_ego_state_tag, image_index = \
                self.concatenate_tokens(
                    cur_input_ids=cur_input_ids, 
                    cur_labels=labels[batch_idx],
                    image_features=image_features[batch_idx],
                    ego_state_features=ego_state_features[batch_idx],
                )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) # [4459 = 3645 + 814, 1024]
            cur_new_labels = torch.cat(cur_new_labels, dim=0) # [4459]
            cur_attention_mask = torch.cat(cur_attention_mask, dim=0)
            cur_ego_state_tag = torch.cat(cur_ego_state_tag, dim=0)

            assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0] == \
                cur_attention_mask.shape[0] == cur_ego_state_tag.shape[0]
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_attention_mask.append(cur_attention_mask)
            ego_state_tag.append(cur_ego_state_tag)
            image_indexes.append(image_index)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        
        # tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # if tokenizer_model_max_length is not None:
        #     for new_input_embed in new_input_embeds:
        #         assert new_input_embed.shape[0] < tokenizer_model_max_length

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_attention_mask[0].device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        ego_state_tag_padded = torch.zeros((batch_size, max_len), dtype=ego_state_tag[0].dtype, device=ego_state_tag[0].device)

        for i, (cur_new_embed, cur_new_labels, cur_attention_mask) in enumerate(zip(new_input_embeds, new_labels, new_attention_mask)):
            cur_len = cur_new_embed.shape[0]
            assert getattr(self.config, "tokenizer_padding_side", "right") == "right"
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask_padded[i, :cur_len] = cur_attention_mask
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                ego_state_tag_padded[i, :cur_len] = ego_state_tag[i]

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask_padded.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, ego_state_tag_padded, aux_losses, image_indexes

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

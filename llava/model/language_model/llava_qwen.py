from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.nn as nn


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.model = LlavaQwenModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if "img_metas" in kwargs: # remove the DataContainer wrapper
            kwargs["img_metas"] = [img_meta.data for img_meta in kwargs["img_metas"]]
        aux_losses = None
        image_indexes = None
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, ego_state_tag, aux_losses, image_indexes) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids, 
                    position_ids=position_ids, 
                    attention_mask=attention_mask, 
                    past_key_values=past_key_values, 
                    labels=labels, # TODO 
                    images=images, 
                    image_sizes=image_sizes,
                    **kwargs,
                )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # [1, 4459, 1024]
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.model.config.is_train:
            if aux_losses:
                hidden_states = output['hidden_states']
                loss_fn = nn.MSELoss()
                image_pred_features = aux_losses['image_pred_features']
                loss_list = []
                for index in image_indexes:
                    loss = loss_fn(hidden_states[:,index[0]+3:index[0]+image_pred_features.shape[1]+3,:], image_pred_features)
                    loss_list.append(loss)
                loss_dict = {'image_pred_loss': sum(loss_list)/len(loss_list)}
                return output, loss_dict
            else:
                return output, None
        else:
            return output



    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, ego_state_tag, aux_losses, image_indexes) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids, 
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                image_sizes=image_sizes,
                **kwargs
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        del kwargs['velocity']
        del kwargs['accel']
        del kwargs['cmd']
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

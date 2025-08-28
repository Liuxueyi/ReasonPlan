import os
import copy
import numpy as np
import pickle
from typing import Dict, Sequence, Mapping
import torch
import transformers
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch.nn.functional as F
from llava.train.config import DataArguments
from llava.constants import *
from llava.mm_utils import tokenizer_token, process_images
from llava import conversation as conversation_lib
from llava.conversation import conv_templates

from mmcv.parallel.data_container import DataContainer
from llava.dataset.b2d_dataset import Bench2DriveDatset
import re

def preprocess_qwen(
        sources, 
        tokenizer: transformers.PreTrainedTokenizer, 
        system_message: str = "You are a helpful assistant."
    ) -> Dict:
    roles = {'Question': "<|im_start|>user", 'Answer': "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens  # '<|im_start|>', '<|im_end|>'
    nl_tokens = "\n"
    _system = "system\n"

    system = im_start + _system + system_message + im_end + nl_tokens
    input_prompt = '' + system

    for sentence in sources:
        if sentence["from"] == "Question":
            question = roles['Question'] + nl_tokens + sentence["value"] + im_end + nl_tokens
            input_prompt += question
            prompt = input_prompt + roles['Answer'] + nl_tokens
        elif sentence["from"] == "Answer":
            answer = roles['Answer'] + nl_tokens + sentence["value"] + im_end + nl_tokens
            input_prompt += answer

    input_ids = tokenizer_token(input_prompt, tokenizer, return_tensors='pt') # 插入特殊token<image>的index -200
    prompt_ids = tokenizer_token(prompt, tokenizer, return_tensors='pt')

    targets = input_ids.clone()
    targets[:len(targets)-len(tokenizer_token(answer, tokenizer))] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,  # 加answer
        prompt_ids=prompt_ids, 
        labels=targets, # 只有answer
    ) 


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 split: str,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.data_args = data_args
        self.tokenizer = tokenizer
        
        self.b2d_dataset = Bench2DriveDatset(data_args)

    def __len__(self):
        return len(self.b2d_dataset)

    @property
    def lengths(self):
        length_list = []
        for sample in len(self):
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        # for sample in self.list_data_dict:
        for i in range(len(self)):
            # cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = 100 # TODO when take prompts with different length as input, fix this
            length_list.append(cur_len)
        return length_list
    

    # NOTE check the dtype of each input data, it's recommended to use float32 instead of float16
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # i = random.randint(100, 200)
        sample_dict = self.b2d_dataset[i]
        
        text_input_question = ""
        text_input_answer = ""
        # data_dict = {"token": source_data_dict['id']}
        data_dict = {"token": str(i)}

        # =============== Action =============== #
        # data_dict['ego_state'] = sample_dict['ego_states'].flatten()
        # data_dict['ego_traj'] = sample_dict['trajectory'][:,:2]

        # =============== CLIP image =============== #
        if self.data_args.use_clip_img_encoder:
            image, image_pred, image_size = self.prepate_clip_data(sample_dict)
            
            data_dict['image'] = image
            data_dict['image_pred'] = image_pred
            data_dict['image_sizes'] = image_size

            if self.data_args.use_text_prompts:
                text_input_question += TEXT_INPUT_CLIP_IMG
            else:
                text_input_question += DEFAULT_IMAGE_TOKEN
        
        # 重新设置问题和答案
        text_question = sample_dict['question']
        text_answer = sample_dict['answer']

        texts = [{'from': 'Question', 'value': text_question}, {'from': 'Answer', 'value': text_answer}]
        data_dict['qas'] = texts
        data_dict['images_path'] = sample_dict['images_path']
        data_dict['velocity'] = sample_dict['velocity']
        data_dict['accel'] = sample_dict['accel']
        data_dict['cmd'] = sample_dict['cmd']
        data_dict.update(preprocess_qwen(texts, self.tokenizer))

        """
        You are the brain of an autonomous vehicle. You're at point (0,0). X-axis is perpendicular and Y-axis is parallel to the direction you're facing.

        Input: 
        - Front view figure <image>.
        - Road structure information centered on ego-vehicle <map>.
        - Navigation information <navigation>.
        - Command information <command>.

        Task: Plan a safe and feasible 8-second driving trajectory at 2 Hz with 16 waypoints. Avoid collisions with other objects.

        - Ego-vehicle trajectory prediction <ego_traj>.
        """
        return data_dict

    def prepate_clip_data(self, source_data_dict):
        # img_metas = source_data_dict["img_metas"].data[3]
        image_list = []
        image_pred_list = []
        for value in source_data_dict['images_path'].values():
            image = Image.open(value).convert('RGB')
            image_list.append(image)
            pre_value = re.sub(r"(\d+)(\.jpg)", lambda m: f"{int(m.group(1)) + 30:05d}{m.group(2)}", value)
            image_pred = Image.open(pre_value).convert('RGB')
            image_pred_list.append(image_pred)
            # break
        # for value in source_data_dict['history_frames']:
        #     if os.path.exists(value):
        #         image = Image.open(value).convert('RGB')
        #         image_list.append(image)
        #     else:
        #         image = Image.new('RGB', (1600, 900), (0, 0, 0))
        #         image_list.append(image)
        processor = self.data_args.image_processor
        image_size = [image.size]
        if self.data_args.image_aspect_ratio == 'anyres':
            image = process_images(image_list, processor, self.data_args)
            image_pred = process_images(image_pred_list, processor, self.data_args)
        else:
            raise ValueError(f"Invalid image aspect ratio: {self.data_args.image_aspect_ratio}")

        return image, image_pred, image_size # image[0].shape [10, 3, 384, 384] [14, 3, 384, 384]


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch_instances = {}
        for key in instances[0].keys():
            batch_instances[key] = [instance[key] for instance in instances]
        
        # parse perception data, remove the DataContainer wrapper
        if 'perception_data' in batch_instances:
            perception_data = batch_instances['perception_data']
            perception_data = self.parse_perception_data(perception_data)
            perception_data['img_metas'] = [DataContainer(img_meta) for img_meta in perception_data['img_metas']] # trick, in avoid of "to device" error
        else:
            perception_data = {}

        # pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_instances['input_ids'],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            batch_instances['prompt_ids'],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch_instances['labels'],
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        ego_traj = torch.stack(batch_instances['ego_traj']) if 'ego_traj' in batch_instances else None

        ego_state = torch.stack(batch_instances['ego_state']) if 'ego_state' in batch_instances else None
        command = torch.stack(batch_instances['command']) if 'command' in batch_instances else None
        
        navigation = torch.stack(batch_instances['navi_data']) if 'navi_data' in batch_instances else None
        navi_mask = torch.stack(batch_instances['navi_mask']) if 'navi_mask' in batch_instances else None
        
        road_pts = torch.stack(batch_instances['road_pts']) if 'road_pts' in batch_instances else None
        lanes_mask = torch.stack(batch_instances['lane_mask']) if 'lane_mask' in batch_instances else None
        points_mask = torch.stack(batch_instances['points_mask']) if 'points_mask' in batch_instances else None
        same_lane = torch.stack(batch_instances['same_lane']) if 'same_lane' in batch_instances else None

        tokens = batch_instances['token'] if 'token' in batch_instances else None
        velocity = torch.stack(batch_instances['velocity']) if 'velocity' in batch_instances else None
        accel = torch.stack(batch_instances['accel']) if 'accel' in batch_instances else None
        cmd = torch.stack(batch_instances['cmd']) if 'cmd' in batch_instances else None
        batch = dict(
            tokens=tokens,
            input_ids=input_ids,
            prompt_ids=prompt_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ego_traj=ego_traj,
            ego_state=ego_state,
            command=command,
            navigation=navigation,
            navi_mask=navi_mask,
            lanes=road_pts,
            lanes_mask=lanes_mask, 
            points_mask=points_mask,
            same_lane=same_lane,
            qas=batch_instances['qas'],
            **perception_data,
            velocity=velocity,
            accel=accel,
            cmd=cmd,
        )

        # stack images
        if instances[0]['image'] is not None:
            if len(instances[0]['image']) == 1:  # only one image per instance
                images = [instance['image'][0] for instance in instances]
                image_sizes = [instance['image_sizes'][0] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images) # images[0].shape torch.Size([10, 3, 384, 384])
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes
        else:
            batch['images'] = None
            batch['image_sizes'] = None
        
        if 'image_pred' in instances[0].keys():
            if instances[0]['image_pred'] is not None:
                if len(instances[0]['image_pred']) == 1:  # only one image per instance
                    images_pred = [instance['image_pred'][0] for instance in instances]
                if all(x is not None and x.shape == images_pred[0].shape for x in images_pred):
                    batch['images_pred'] = torch.stack(images_pred) # images[0].shape torch.Size([10, 3, 384, 384])
                else:
                    batch['images_pred'] = images_pred
            else:
                batch['images_pred'] = None
        
        if 'images_path' in batch_instances:
            batch['images_path'] = batch_instances['images_path']

        return batch # batch['images'].shape torch.Size([1, 10, 3, 384, 384])

    
    def stack_perception_data(self, batch):
        """Puts each data field into a tensor/DataContainer with outer dimension
        batch size.

        Extend default_collate to add support for
        :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

        1. cpu_only = True, e.g., meta data
        2. cpu_only = False, stack = True, e.g., images tensors
        3. cpu_only = False, stack = False, e.g., gt bboxes
        """
        if not isinstance(batch, Sequence):
            raise TypeError(f'{batch.dtype} is not supported.')
        
        samples_per_gpu = len(batch)

        if isinstance(batch[0], DataContainer):
            stacked = []
            if batch[0].cpu_only:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
                return DataContainer(
                    stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
            elif batch[0].stack:
                for i in range(0, len(batch), samples_per_gpu):
                    assert isinstance(batch[i].data, torch.Tensor)

                    if batch[i].pad_dims is not None:
                        ndim = batch[i].dim()
                        assert ndim > batch[i].pad_dims
                        max_shape = [0 for _ in range(batch[i].pad_dims)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = batch[i].size(-dim)
                        for sample in batch[i:i + samples_per_gpu]:
                            for dim in range(0, ndim - batch[i].pad_dims):
                                assert batch[i].size(dim) == sample.size(dim)
                            for dim in range(1, batch[i].pad_dims + 1):
                                max_shape[dim - 1] = max(max_shape[dim - 1],
                                                        sample.size(-dim))
                        padded_samples = []
                        for sample in batch[i:i + samples_per_gpu]:
                            pad = [0 for _ in range(batch[i].pad_dims * 2)]
                            for dim in range(1, batch[i].pad_dims + 1):
                                pad[2 * dim -
                                    1] = max_shape[dim - 1] - sample.size(-dim)
                            padded_samples.append(
                                F.pad(
                                    sample.data, pad, value=sample.padding_value))
                        stacked.append(torch.stack(padded_samples))
                    elif batch[i].pad_dims is None:
                        stacked.append(
                            torch.stack([
                                sample.data
                                for sample in batch[i:i + samples_per_gpu]
                            ]))
                    else:
                        raise ValueError(
                            'pad_dims should be either None or integers (1-3)')

            else:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
        elif isinstance(batch[0], Sequence):
            transposed = zip(*batch)
            return [self.stack_perception_data(samples) for samples in transposed]
        elif isinstance(batch[0], Mapping):
            return {
                key: self.stack_perception_data([d[key] for d in batch])
                for key in batch[0]
            }
        else:
            return torch.stack(batch)

    def parse_perception_data(self, batch):
        batch = self.stack_perception_data(batch)
        data_dict = {}
        for key, value in batch.items():
            if isinstance(value, DataContainer):
                data_dict[key] = value.data[0]
            elif isinstance(value[0], DataContainer):
                data_dict[key] = value[0].data
            else:
                data_dict[key] = value
        return data_dict


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_args=data_args,
                                split='train')
    # eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_args=data_args,
    #                             split='val')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                # eval_dataset=eval_dataset,
                data_collator=data_collator)

import os
import pickle
import json
import random
import torch
import copy
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from mmcv.datasets import build_dataset
from llava.utils import rank0_print

def normalize_angle(angle):
    return angle % 360

def sample_from_space(X):
    return X[torch.randint(0, X.size(0), (1,))]

def tensor_to_str(traj_tensor):
    traj_str = ""
    for i, traj in enumerate(traj_tensor):
        if i == traj_tensor.size(0) - 1:
            traj_str += f"({traj[0].item():.2f}, {traj[1].item():.2f})"
        else:
            traj_str += f"({traj[0].item():.2f}, {traj[1].item():.2f}), "
    return traj_str

def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


class Bench2DriveDatset(Dataset):

    def __init__(self, data_args) -> None:
        super(Bench2DriveDatset, self).__init__()

        self.qas_dir_path = data_args.text_data_path
        self.data_root = data_args.b2d_root
        self.history_frames = data_args.history_frames
        self.future_frames = data_args.future_frames
        self.sample_rate = 5
        self.data_infos = pickle.load(open(self.qas_dir_path, 'rb'))

        self.train_stage = data_args.train_stage
        self.reasoning_enable = data_args.reasoning_enable

        rank0_print(f"Loaded {len(self.data_infos)} samples from {self.qas_dir_path}")
        rank0_print(f"Number of valid samples: {len(self.data_infos)}")

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        data_dict = {}
        text_question, text_answer, image_path, history_frames_path, velocity, accel, cmd = self.set_prompt(index)
        for key, value in image_path.items():
            image_path[key] = os.path.join(self.data_root, value)
        

        history_frames_path = [os.path.join(self.data_root, path) for path in history_frames_path]
        
        data_dict['question'] = text_question
        data_dict['answer'] = text_answer
        data_dict['images_path'] = image_path
        data_dict['history_frames'] = history_frames_path
        data_dict['velocity'] = velocity
        data_dict['accel'] = accel
        data_dict['cmd'] = cmd
        return data_dict
    
    def set_prompt(self, index):
        try:
            CoT_question, CoT_answer, image_path, history_frames_path, velocity, accel, cmd  = self.get_QA_data(index)
        except:
            CoT_question = ''
            CoT_answer = ''
            image_path = ''
            history_frames_path = ''
            velocity = torch.tensor([0, 0])
            accel = torch.tensor([0, 0])
            cmd = torch.tensor(0)
            print(f"QA data not found for index {index}")

        return CoT_question, CoT_answer, image_path, history_frames_path, velocity, accel, cmd 


    def get_valid_indices(self):
        """
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3
        LANEFOLLOW = 4
        CHANGELANELEFT = 5
        CHANGELANERIGHT = 6
        """
        valid_indices = []
        cmd_idx_dict = {}
        for idx in tqdm(range(len(self.data_infos)), desc="Filtering valid samples"):
            adj_idx_list = list(range(idx-self.history_frames*self.sample_rate,idx+(self.future_frames+1)*self.sample_rate,self.sample_rate))

            if (adj_idx_list[0]<0) or (adj_idx_list[-1]>=len(self.data_infos)):
                continue
            
            num = 0
            for j in adj_idx_list:
                if self.data_infos[j]['folder'] == self.data_infos[idx]['folder']:
                    num += 1

            if num == len(adj_idx_list):
                valid_indices.append(idx)
                command = self.data_infos[idx]['command_near']
                if command not in cmd_idx_dict:
                    cmd_idx_dict[command] = [idx]
                else:
                    cmd_idx_dict[command].append(idx)
        return valid_indices, cmd_idx_dict
    
    def get_QA_data(self, index):
        qa_data = self.data_infos[index]
        image_path = qa_data['image_path']
        history_frames_path = qa_data['history_frames']
    
        if self.train_stage == 'first_align':
            question = qa_data['description_prompt']
            answer = qa_data['description']
        elif self.train_stage == 'second_finetune':
            question = qa_data['system_prompt']
            if self.reasoning_enable:
                answer = qa_data['answer']
            else:
                answer = qa_data['traj_answer']

        velocity = torch.tensor(qa_data['velocity_xy'])
        accel = torch.tensor(qa_data['accel_xy'])  
        cmd = torch.tensor(qa_data['cmd'])
        cmd = self.command2hot(cmd)
        return question, answer, image_path, history_frames_path, velocity, accel, cmd
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return torch.from_numpy(cmd_one_hot).float()

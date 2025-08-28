import copy
import torch
import ast
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from team_code.carla_map_utils import clip_map_participant, get_format_output, StateSE2
from llava.dataset.dataset import DataCollatorForSupervisedDataset
from llava.dataset.dataset import preprocess_qwen
from llava.dataset.b2d_dataset import tensor_to_str, COMMAND_DICT
from llava.mm_utils import process_images
from llava.constants import (
    TEXT_INPUT_CLIP_IMG,
    TEXT_INPUT_PERCEPTION,
    TEXT_INPUT_OBJ,
    TEXT_ANSWER_EGO_TRAJ,
    TEXT_INPUT_HEAD,
    TEXT_TASK,
    TEXT_PROMPT,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PERCEPTION_TOKEN,
    DEFAULT_OBJ_TOKEN,
    DEFAULT_EGO_TRAJ_TOKEN,
    TEXT_INPUT_MAP,
    DEFAULT_MAP_TOKEN,
    TEXT_INPUT_NAVI,
    DEFAULT_NAVI_TOKEN,
    TEXT_INPUT_CMD,
    DEFAULT_CMD_TOKEN,
)


class DatasetUtils:
    def __init__(self, config, tokenizer, torch_dtype):
        self.data_args = config
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorForSupervisedDataset(tokenizer)
        self.device="cuda"
        self.torch_dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float32

    def get_input_data(self, sample_dict):
        data_dict = {}

        # =============== CLIP image =============== #
        if self.data_args.use_clip_img_encoder:
            image, image_size = self.prepate_clip_data(sample_dict)
            
            data_dict['image'] = image
            data_dict['image_sizes'] = image_size

        # 重新设置问题和答案
        text_question = self.set_prompt(sample_dict)

        texts = [
            {'from': 'Question', 'value': text_question}, 
            {'from': 'Answer', 'value': 'None'}]
        data_dict['qas'] = texts
        data_dict.update(preprocess_qwen(texts, self.tokenizer))

        input_data_batch = self.collate_fn([data_dict])
        
        input_data = {
            'images': input_data_batch['images'],
            'image_sizes': input_data_batch['image_sizes'],
            'input_ids': input_data_batch['prompt_ids'],
        }
        input_data = self.put_on_device(input_data)

        return input_data, text_question
    
    def decode_tokens(self, outputs):
        string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        traj = string[0].split('planning trajectory should be:')[-1].strip('.')
        traj = np.array(ast.literal_eval(f"[{traj}]"))
        return traj, string[0]
    
    def put_on_device(self, input_data_batch):
        for key, data in input_data_batch.items():
            if key != 'img_metas' and data is not None:
                if torch.is_tensor(data):
                    if data.dtype in [torch.float32, torch.float64, torch.float16]:
                        input_data_batch[key] = input_data_batch[key].to(device=self.device, dtype=self.torch_dtype)
                    else:
                        input_data_batch[key] = input_data_batch[key].to(device=self.device)
                elif isinstance(data, list) and torch.is_tensor(data[0]):
                    if data[0].dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                        input_data_batch[key][0] = input_data_batch[key][0].to(device=self.device, dtype=self.torch_dtype)
                    else:
                        input_data_batch[key][0] = input_data_batch[key][0].to(device=self.device)
        
        return input_data_batch

    def set_prompt(self, sample_dict): 
        command = COMMAND_DICT[sample_dict['command_near']]
        prompt = f"This is a picture taken by a car camera <image>. Please describe the objects that are closely related to the vehicle's movement in the photo, For example, what is the status of the traffic lights, and do you need to be aware of pedestrians and vehicles, and are there any obstacles ahead, Current driving command is {command}, If you were driving, what would you do next?"

        hist_traj_str = tensor_to_str(sample_dict['trajectory'])

        # prompt += f"\nThe current vehicle speed is {sample_dict['velocity']} m/s, the throttle is {sample_dict['throttle']}, the brake is {sample_dict['brake']}, the steer is {sample_dict['steer']}, the history trajectory is {hist_traj_str} and the current target point is {tensor_to_str(sample_dict['command_far_xy'])}, Please predict the 8s future trajectory."
        prompt += f"\nThe current target point is {tensor_to_str(sample_dict['command_far_xy'])} and the history trajectory is {hist_traj_str}, please predict the 8s future trajectory."

        return prompt

    def prepate_clip_data(self, sample_dict):
        image = sample_dict['CAM_FRONT']
        image = Image.fromarray(image).convert('RGB')
        
        processor = self.data_args.image_processor
        image_size = [image.size]
        if self.data_args.image_aspect_ratio == 'anyres':
            image = process_images([image], processor, self.data_args)
        else:
            raise ValueError(f"Invalid image aspect ratio: {self.data_args.image_aspect_ratio}")

        return image, image_size


    def resample_trajectory(self, points, num_points):
        # 计算原始点之间的累积距离
        distances = np.sqrt(np.sum(np.diff(points[:, :2], axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        # 创建新的均匀分布的距离
        new_distances = np.linspace(0, cumulative_distances[-1], num_points)
        
        # 对每个坐标进行插值
        interp_func_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
        interp_func_y = interp1d(cumulative_distances, points[:, 1], kind='linear')
        # interp_func_z = interp1d(cumulative_distances, points[:, 2], kind='linear')
        
        new_points_x = interp_func_x(new_distances)
        new_points_y = interp_func_y(new_distances)
        # new_points_z = interp_func_z(new_distances)
        new_points = np.vstack((new_points_x, new_points_y)).T
        
        return new_points

    def get_path(self, navi_points):
        # 第一个点为当前点，向后面取16个点，80米的导航轨迹
        traj = [navi_points[0].reshape(1, 2)]
        dist = 0
        for i in range(len(navi_points)-1):
            if dist >= 80:
                break
            traj.append(navi_points[i+1].reshape(1, 2))
            dist += np.linalg.norm(traj[-1] - traj[-2])
        traj = np.array(traj).reshape(-1, 2)
        if dist > 80:
            navi_path = self.resample_trajectory(traj, 16)
            navi_valid = np.ones(16, dtype=bool)
        else:
            future = int(dist/5)
            if future == 0:
                navi_path = np.tile(navi_points[-1], (16,1))
                navi_valid = np.zeros(16, dtype=bool)
            else:
                navi_path_ = self.resample_trajectory(traj, future)
                navi_path = np.tile(navi_path_[-1], (16,1))
                navi_path[:future] = navi_path_
                navi_valid = np.zeros(16, dtype=bool)
                navi_valid[:future] = True

        return navi_path, navi_valid

    def get_navi_path(self, ego_pose, route):

        routes = [r[0] for r in route]
        routes = np.array(routes)
        routes[:,1] = -routes[:,1]
        dist = np.linalg.norm(routes - np.array(ego_pose['pos'])[:2].reshape(1, 2), axis=1)
        closest_idx = np.argmin(dist)
        routes = routes[closest_idx:]
        navi_path, valid = self.get_path(routes)

        lidar2global_cur = ego_pose["lidar2global"]
        
        navi_points = np.zeros((16,4))
        navi_points[:,0:2] = navi_path
        navi_points[:,-1] = 1

        navi_points = np.dot(np.linalg.inv(lidar2global_cur), navi_points.T).T[:,:2]

        return navi_points, valid

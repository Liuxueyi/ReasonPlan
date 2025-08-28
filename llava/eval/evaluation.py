import os
import pickle
import json
import torch
import math
import shutil
import pandas as pd
import numpy as np
import transformers
from tqdm import tqdm
import warnings
from easydict import EasyDict
warnings.filterwarnings("ignore")
from mmcv import Config
from safetensors.torch import load_file as safe_load_file
from dataclasses import dataclass, field
from llava.model.builder import load_pretrained_model
from llava.dataset.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def batch_to_device(batch, device, dtype=torch.float32):
    float_type = [torch.float16, torch.float32, torch.float64, torch.int8, torch.bfloat16]
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            tensor_type = batch[key].dtype if batch[key].dtype not in float_type else dtype
            batch[key] = batch[key].to(device=device, dtype=tensor_type)
        elif isinstance(batch[key], list):
            if isinstance(batch[key][0], torch.Tensor):
                tensor_type = batch[key][0].dtype if batch[key][0].dtype not in float_type else dtype
                batch[key] = [x.to(device=device, dtype=tensor_type) for x in batch[key]]
            elif isinstance(batch[key][0], list):
                if isinstance(batch[key][0][0], torch.Tensor):
                    tensor_type = batch[key][0][0].dtype if batch[key][0][0].dtype not in float_type else dtype
                    batch[key] = [[x.to(device=device) for x in y] for y in batch[key]]
    return batch


def save_results(results, save_path):
    if save_path is not None:
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                results = json.dumps(results, indent=2)
                f.write(results)
                print(f"Save to {save_path}")
        elif save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
                print(f"Save to {save_path}")
        else:
            raise ValueError("Unsupported file format. Please use either .json or .pkl.")

def update_args(args, model):
    args.image_grid_pinpoints = model.config.image_grid_pinpoints
    args.image_aspect_ratio = model.config.image_aspect_ratio
    args.image_processor = model.get_model().vision_tower.image_processor
    args.use_clip_img_encoder = model.config.use_clip_img_encoder
    args.use_map_encoder = model.config.use_map_encoder
    args.use_perception_encoder = model.config.use_perception_encoder
    args.use_navi_encoder = model.config.use_navi_encoder
    args.use_command_encoder = model.config.use_command_encoder
    args.use_text_prompts = model.config.use_text_prompts
    args.perception_config = EasyDict(model.config.perception_config)

    return args


def eval_model(args):
    # =============== Set up model and tokenizer =============== #
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_name, args.torch_dtype)

    model.cuda()
    # =============== Set up model and tokenizer =============== #

    # =============== Set up data module =============== #
    args = update_args(args, model) 

    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=args, split='val')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    )
    # =============== Set up data module =============== #

    # =============== inference =============== #
    model.eval()
    results = dict()
    # fig_filedir = '/data/wangjl/Air/Bench2Drive/LLava-Next-Nuscenes/output/map/'
    # os.makedirs(fig_filedir, exist_ok=True)
    # command_dict = {
    #     0: 'chaneg_lane_left',
    #     1: 'chaneg_lane_right',
    #     2: 'lane_follow',
    #     3: 'left',
    #     4: 'right',
    #     5: 'straight',
    # }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
            if batch is None:
                continue
            batch = batch_to_device(batch, device=model.device, dtype=model.dtype)
            try:
                pred_traj = model.forward(**batch,)
            except Exception as e:
                print(e)
                print(batch['tokens'])
                continue

            for i, token in enumerate(batch['tokens']):
                results[token] = {
                            'gt': batch['ego_traj'][i].detach().cpu().float().numpy(),
                            'output': pred_traj[i].detach().cpu().float().numpy()
                            }
                
                # import matplotlib.pyplot as plt
                # hist = batch['ego_state'][i].reshape(2, 8)[:,:2].float().cpu().numpy()
                # # heading = batch['ego_state'][i].reshape(2, 8)[-1,2].item()
                # fut = batch['ego_traj'][i].float().cpu().numpy()
                # pred = pred_traj[i].float().cpu().numpy()
                # navi = batch['navigation'][i].float().cpu().numpy()
                # road_pts = batch['lanes'][i].float().cpu().numpy()
                # point_mask = batch['points_mask'][i].bool().cpu().numpy()
                # road = road_pts[point_mask]
                # command = batch['command'][i].float().cpu().numpy()
                # command = command_dict[np.argmax(command)]
                # plt.scatter(road[::5,0], road[::5,1], c='k', s=1, label='road')
                # plt.scatter(navi[:,0], navi[:,1], c='y', s=15, label='navi')
                # # plt.scatter(hist[:,0], hist[:,1], c='g', s=10, label='hist')
                # plt.plot(hist[:,0], hist[:,1], 'go-', label='hist')
                # # plt.scatter(fut[:,0], fut[:,1], c='r', s=10, label='fut')
                # plt.plot(fut[:,0], fut[:,1], 'ro-', label='fut')
                # # plt.scatter(pred[:,0], pred[:,1], c='b', s=10, label='pred')
                # plt.plot(pred[:,0], pred[:,1], 'bo-', label='pred')
                # plt.text(0, 0, f'command:{command}', fontsize=12, color='k')
                # plt.gca().set_aspect(1)
                # plt.legend()
                # plt.xlim(-100, 100)
                # # plt.ylim(-10, 10)
                # plt.savefig(os.path.join(fig_filedir, f'{token}.png'))
                # print(f'save {token}')
                # plt.close()
                # # break

    folder, filename = os.path.split(args.save_path)
    suffix = filename.split('.')[-1]
    temp_folder = os.path.join(folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    
    save_path = os.path.join(temp_folder, f'{args.chunk_idx}.{suffix}') 
    save_results(results, save_path)
    # =============== inference =============== #

    return temp_folder


def get_matrics(results, save_path):
    if save_path.endswith(".json"):
        save_path = save_path.replace('.json', '_metrics.csv')
    elif save_path.endswith(".pkl"):
        save_path = save_path.replace('.pkl', '_metrics.csv')
    
    metric_table = []
    
    for token, result in results.items():
        gt = result['gt']
        output = result['output']
        l2 = np.linalg.norm(gt[:,:2] - output[:,:2], axis=-1)
        l2_2s = l2[:4].mean()
        l2_uniad = l2[1::2]
        row = {'token': token, 'Uniad_L2_Avg': l2_uniad.mean(), 'l2_2s': l2_2s}
        for i in range(l2_uniad.shape[0]):
            row[f'Uniad_L2_{i+1}s'] = l2_uniad[i]
        metric_table.append(row)

    metric_table = pd.DataFrame(metric_table)
    average_row = metric_table.drop(columns=["token"]).mean(skipna=True)
    average_row["token"] = "average"
    metric_table.loc[len(metric_table)] = average_row
    metric_table.to_csv(save_path, index=False)
    print(f"Metrics are saved to {save_path}")


def merge_results(args, temp_folder, output_files):
    results = dict()
    for file in output_files:
        if file.endswith(".json"):
            with open(os.path.join(temp_folder, file), "r") as temp_f:
                result = json.load(temp_f)
                results.update(result)
        elif file.endswith(".pkl"):
            with open(os.path.join(temp_folder, file), "rb") as temp_f:
                result = pickle.load(temp_f)
                results.update(result)
        else:
            raise ValueError("Unsupported file format. Please use either .json or .pkl.")
    save_results(results, args.save_path)
    shutil.rmtree(temp_folder)
    print(f"All results are saved to {args.save_path}")
    get_matrics(results, args.save_path)


def main(args):
    temp_folder = eval_model(args)

    output_files = os.listdir(temp_folder)
    if len(output_files) == args.num_chunks:
        merge_results(args, temp_folder, output_files)


@dataclass
class EvalArguments:
    torch_dtype: str = "bf16"

    # model
    model_path: str = field(default='/data/wangjl/Air/Bench2Drive/LLava-Next-Nuscenes/checkpoints/bench2drive_base_qwen1p5_nomap/checkpoint-5126')
    model_name: str = field(default="llava_qwen_1_5-0.5B")

    # data
    # image_folder: str = field(default='/data/wangjl/Air/navsim_workspace')
    # eval_data_path: str = field(default=None)
    map_action_path: str = field(default='/data/wangjl/Air/Bench2Drive/data/bench2drive/centerlines')
    # perception_config_path: str = field(default='/data/wangjl/Air/Bench2Drive/LLava-Next-Nuscenes/Bench2DriveZoo/adzoo/apollofm_perception/configs/VAD/VAD_base_e2e_b2d.py')
    # navi_data_path: str = field(default='/data/wangjl/Air/Bench2Drive/data/bench2drive/navigation')

    # evaluation
    batch_size: int = field(default=5)
    num_workers: int = field(default=5)
    num_chunks: int = field(default=1)
    chunk_idx: int = field(default=0)
    
    # output
    save_path: str = field(default='')


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(EvalArguments)
    args = parser.parse_args()
    main(args)
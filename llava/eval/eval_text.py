import torch
import os
import ast
import json
from tqdm import tqdm
import numpy as np
import transformers
import warnings
warnings.filterwarnings("ignore")
from llava.dataset.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from llava.train.config import EvaluationArguments
from llava.model.builder import load_pretrained_model

import numpy as np

import os
import json
import argparse

def update_args(eval_args, model_config, image_processor):
    eval_args.image_processor = image_processor
    eval_args.mm_use_im_start_end = model_config.mm_use_im_start_end
    eval_args.image_aspect_ratio = model_config.image_aspect_ratio
    eval_args.image_grid_pinpoints = model_config.image_grid_pinpoints
    eval_args.image_crop_resolution = model_config.image_crop_resolution
    eval_args.image_split_resolution = model_config.image_split_resolution
    eval_args.history_frames = model_config.history_frames
    eval_args.future_frames = model_config.future_frames
    eval_args.use_clip_img_encoder = model_config.use_clip_img_encoder

    return eval_args

def filter_and_decode(outputs, tokenizer, skip_value=-100):
    # 过滤掉指定的 skip_value
    filtered_outputs = [[token for token in sequence if token != skip_value] for sequence in outputs]
    # 执行解码
    return tokenizer.batch_decode(filtered_outputs, skip_special_tokens=True)

def main(eval_args):
    device = 'cuda'
    tokenizer, model, image_processor, context_len = load_pretrained_model(eval_args.checkpoint_path, attn_implementation=None)
    model = model.to(device)
    model.eval()
    
    # update eval_args
    eval_args = update_args(eval_args, model.config, image_processor)
    model.config.is_train = eval_args.is_train

    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, split='val', data_args=eval_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=1,
        num_workers=1,
        drop_last=False,
        shuffle=False
    )

    os.makedirs(eval_args.output_dir, exist_ok=True)
    l2_error = {}
    cap_list = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        prompt_ids = batch["prompt_ids"].to(device)
        qas = batch["qas"]
        images = [i.to(device) for i in batch["images"]]
        image_sizes = batch["image_sizes"]
        tokens = batch["tokens"]
        ego_traj = qas[0][1]['value'].split("planning trajectory should be:")[-1].strip('.')
        ego_traj = np.array(ast.literal_eval(f"[{ego_traj}]"))
        ego_traj = np.array(ego_traj)
        images_path = batch['images_path']
        velocity = batch['velocity'].to(device)
        accel = batch['accel'].to(device)
        cmd = batch['cmd'].to(device)
        with torch.inference_mode(), torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(
                input_ids=prompt_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=1024,
                top_k=1,
                velocity=velocity,
                accel=accel,
                cmd=cmd
                )

        # decode the output
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        for i, token in enumerate(tokens):
            question_answer = {
                "question": qas[i][0]['value'],
                "answer": qas[i][1]['value'],
                "predicted_answer": output[i],
                "images_path": images_path[i],
            }
            try:
                traj = output[i].split("planning trajectory should be:")[-1].strip('.')
                traj = np.array(ast.literal_eval(f"[{traj}]"))
                traj = np.array(traj)
                error = np.linalg.norm(traj - ego_traj, axis=-1)
                l2_error[token] = error

                question_answer["planning_trajectory"] = traj.tolist()
                question_answer["error"] = error.tolist()
                generated_texts = question_answer["predicted_answer"]
                reference_texts = question_answer["answer"]
                cap_list.append([generated_texts, reference_texts])
                
            except:
                print(f"Failed to parse planning trajectory for {token}")
            with open(os.path.join(eval_args.output_dir, f"sample_{token}.json"), "w") as f:
                json.dump(question_answer, f, indent=4)
        
    with open(os.path.join(eval_args.output_dir, "l2_error.txt"), "w") as f:
        f.write(f"Mean L2 error: {np.mean(list(l2_error.values()))}\n")
        mean_l2 = np.array(list(l2_error.values())).mean(axis=0)
        f.write(f"Mean L2 error for each frame: {mean_l2.tolist()}\n")
        l2_2s = mean_l2[:4].mean()
        f.write(f"2s mean L2 error: {l2_2s}\n")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(EvaluationArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    main(eval_args)
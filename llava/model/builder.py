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
import os
import glob
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from llava.utils import rank0_print
from llava import conversation as conversation_lib
from llava.mm_utils import smart_tokenizer_and_embedding_resize
import json

def load_checkpoint(checkpoint_path):
    checkpoints = {}
    all_keys = {}
    
    safetensors_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors")) 
    for safetensors_file in safetensors_files:
        file_keys = []
        with safe_open(safetensors_file, framework="pt") as f:
            for k in f.keys():
                file_keys.append(k)
                if k == "model.image_newline":
                    continue
                checkpoints[k] = f.get_tensor(k)

        safetensors_filename = os.path.basename(safetensors_file)
        all_keys[safetensors_filename] = file_keys
                
    # 将所有键保存到一个JSON文件中，文件名为 model.safetensors.index
    index_file_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_file_path, "w") as index_file:
        json.dump(all_keys, index_file, indent=4)

    print(f"All keys saved to {index_file_path}")


    if 'lm_head.weight' not in checkpoints:
        checkpoints['lm_head.weight'] = checkpoints['model.embed_tokens.weight']

    return checkpoints


def load_pretrained_model(model_path, model_name="llava_qwen", torch_dtype="bf16", device_map='auto', attn_implementation="flash_attention_2", **kwargs):
    rank0_print(f"Loaded LLaVA model: {model_path}")
    if "qwen" in model_name.lower():
        from llava.model import LlavaQwenForCausalLM as LlavaForCausalLM
    else:
        raise ValueError(f"Model {model_name} not supported")

    torch_dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float32
    model = LlavaForCausalLM.from_pretrained(
        model_path,
        cache_dir=None,
        # device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="right", 
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    
    rank0_print(f"Model Class: {model.__class__.__name__}")

    image_processor = None
    if getattr(model.config, "use_clip_img_encoder", True):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        vision_tower.to(device="cuda", dtype=torch_dtype)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048
    
    if "llama" in model_name.lower():
        from llava.conversation import conv_llava_llama_3 as conv_llava
        conv_llava.tokenizer_id = model.config.name_or_path
        conv_llava.tokenizer = tokenizer
    elif "qwen" in model_name.lower():
        from llava.conversation import conv_qwen as conv_llava
    else:
        raise ValueError(f"Unknown version: {model_name}")
    conversation_lib.default_conversation = conv_llava

    checkpoints = load_checkpoint(model_path)
    model.load_state_dict(checkpoints)
    rank0_print(f"Loaded model from {model_path}")

    return tokenizer, model, image_processor, context_len

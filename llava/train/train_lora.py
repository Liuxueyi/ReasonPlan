import os
import logging
import pathlib
import torch
import transformers
import tokenizers

from llava.mm_utils import smart_tokenizer_and_embedding_resize
from llava.train.config import ModelArguments, DataArguments, TrainingArguments
from llava.dataset.dataset import make_supervised_data_module
from llava.train.llava_trainer import LLaVATrainer
from llava import conversation as conversation_lib
from llava.utils import save_model_and_args, rank0_print

import warnings
warnings.filterwarnings("ignore")

from transformers import TrainerCallback, TrainerControl, TrainerState

import os
import time
from transformers import TrainerCallback, TrainerControl, TrainerState

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

local_rank = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train(attn_implementation=None):
    # =============== Parse arguments =============== #
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # =============== Parse arguments =============== #

    train_stage = data_args.train_stage
    if train_stage == 'first_align':
        model_args.freeze_backbone = True
        model_args.freeze_vision_tower = True
        model_args.tune_mm_mlp_adapter = True
    elif train_stage == 'second_finetune':
        model_args.freeze_backbone = False
        model_args.freeze_vision_tower = True
        model_args.tune_mm_mlp_adapter = True
    else:
        raise ValueError("Training stage error")
    # =============== Training stage =============== #

    # =============== Set up Llava bachbone =============== #
    from llava.model import LlavaQwenForCausalLM as LlavaForCausalLM
    model = LlavaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # =============== Set up tokenizer =============== #
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    from llava.conversation import conv_qwen as conv_llava
    conversation_lib.default_conversation = conv_llava

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    # =============== Set up tokenizer =============== #

    # =============== Set up clip vision module =============== #
    model.config.use_clip_img_encoder = data_args.use_clip_img_encoder = model_args.use_clip_img_encoder
    # print('******************')
    # print('use_clip_img_encoder:', model_args.use_clip_img_encoder)
    if model_args.use_clip_img_encoder:
        # initialize vision modules
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_projector_lr = training_args.mm_projector_lr

        # set up vision tower and get image processor
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor

        model.config.freeze_vision_tower = model_args.freeze_vision_tower
        if model_args.freeze_vision_tower:
            vision_tower.requires_grad_(False)
        else:
            vision_tower.requires_grad_(True)
        # NOTE freeze all param except mm_projector, which is used for feature alignment
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # add image special tokens
        model.config.mm_use_im_start_end = training_args.use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        model.get_model().del_vision_modules()
    # =============== Set up clip vision module =============== #

    # =============== Set up other configs =============== #
    model.config.use_text_prompts = data_args.use_text_prompts
    model.config.image_folder = data_args.image_folder
    model.config.history_frames = data_args.history_frames
    model.config.future_frames = data_args.future_frames
    model.config.text_data_path = data_args.text_data_path
    model.config.is_train = training_args.is_train
    model.config.nsp_enable = training_args.nsp_enable
    model.config.only_front = training_args.only_front
    model.config.im_weight = training_args.im_weight

    # =============== Set up other configs =============== #

    # =============== Save requires_grad.txt =============== #
    for p in model.get_model().ego_vel_mlp.parameters():
        p.requires_grad = True
    for p in model.get_model().ego_acc_mlp.parameters():
        p.requires_grad = True
    for p in model.get_model().command_mlp.parameters():
        p.requires_grad = True
    save_model_and_args(model, training_args, model_args, data_args)
    # =============== Save requires_grad.txt =============== #

    # =============== Set up data module =============== #
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    rank0_print("Set up data module done.")
    # =============== Set up data module =============== #

    # =============== Set up trainer and train =============== #
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint")  
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # =============== Set up trainer and train =============== #

    # =============== Save model =============== #
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler", "ego_vel_mlp", "ego_acc_mlp", "command_mlp"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

if __name__ == "__main__":
    train()

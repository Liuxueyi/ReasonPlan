from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")

    # =============== Quantization =============== #
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)

    # =============== Training =============== #
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=12)
    dataloader_num_workers: int = field(default=8)
    
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=2e-6)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    gradient_accumulation_steps: int = field(default=1)
    
    gradient_checkpointing: bool = field(default=True)

    mm_vision_tower_lr: Optional[float] = 5e-7
    mm_projector_lr: Optional[float] = None

    # =============== Evaluation =============== #
    evaluation_strategy: str = field(default="no")

    # =============== FM =============== #
    model_max_length: int = field(
        default=8192,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    group_by_modality_length: bool = field(default=True, metadata={"help": "Group the input data by modality length."})

    # =============== Log =============== #
    save_strategy: str = field(default="epoch")
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=10)

    logging_steps: int = field(default=1000)
    # report_to: str = field(default="wandb")
    report_to: str = field(default="none")
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    is_train: bool = True
    nsp_enable: bool = False
    only_front: bool = False
    im_weight: float = 1.0
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

@dataclass
class DataArguments:
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'anyres'
    image_crop_resolution: Optional[int] = 224
    image_grid_pinpoints = [[336,672],[672,336],[672,672],[1008,336],[336,1008]]

    # =============== Text Data =============== #
    use_text_prompts: bool = field(default=True, metadata={"help": "Use text prompts for the training."})

    # =============== Bench2Drive Data =============== #
    b2d_root: str = "data/bench2drive"

    # =============== Action Data =============== #
    train_data_path: str = field(default=None,metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None,metadata={"help": "Path to the evaluation data."})

    # =============== Text Data =============== #
    text_data_path: Optional[str] = field(default=None)
    reasoning_enable: bool = False
    train_stage: str = 'none'
    history_frames: int = 4
    future_frames: int = 6

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llama3-llava-next-8b")
    freeze_backbone: bool = field(default=False)
    
    # =============== CLIP Img Encoder =============== #
    use_clip_img_encoder: bool = field(default=True)

    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    freeze_vision_tower: bool = field(default=True)
    
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Freeze all param except mm_projector, which is used for feature alignment."})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)

    mm_patch_merge_type: Optional[str] = field(default='spatial_unpad')
    
    mm_use_im_start_end: bool = field(default=False) # No use by now
    mm_use_im_patch_token: bool = field(default=False) # No use by now



@dataclass
class EvaluationArguments:
    is_multimodal: bool = True
    image_aspect_ratio: str = 'anyres'
    image_crop_resolution: Optional[int] = 224
    image_grid_pinpoints = [[336,672],[672,336],[672,672],[1008,336],[336,1008]]

    # =============== Text Data =============== #
    use_text_prompts: bool = field(default=True, metadata={"help": "Use text prompts for the training."})

    # =============== Bench2Drive Data =============== #
    b2d_root: str = "data/bench2drive"

    # =============== Action Data =============== #
    checkpoint_path: str = field(default=None, metadata={"help": "Path to the checkpoint."})
    output_dir: str = field(default=None, metadata={"help": "Path to save the evaluation results."})

    # =============== Text Data =============== #
    text_data_path: Optional[str] = field(default=None)
    is_train: bool = False
    reasoning_enable: bool = False
    train_stage: str = 'second_finetune'
    history_frames: int = 4
    future_frames: int = 6
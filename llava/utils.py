import datetime
import logging
import logging.handlers
import os
import sys
import json
from prettytable import PrettyTable
from deepspeed import zero

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def save_model_and_args(model, training_args, model_args, data_args):
    if training_args.local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        model_stat_path = 'model_stat'
        os.makedirs(os.path.join(training_args.output_dir, model_stat_path), exist_ok=True)
        with open(f'{training_args.output_dir}/{model_stat_path}/requires_grad.txt', 'w') as f:
            for name, param in model.named_parameters():
                f.write(f"{name} {param.requires_grad}\n")
            print(f"requires_grad saved to {training_args.output_dir}")
        # with open(f'{training_args.output_dir}/{model_stat_path}/num_params.txt', 'w') as f:
        #     with zero.GatheredParameters(model.parameters()):
        #         table = PrettyTable(["Model", "params"])
        #         table.add_row(["model", sum(p.numel() for p in model.parameters())])
        #         table.add_row(["trainable", sum(p.numel() for p in model.parameters() if p.requires_grad)])
        #         table.add_row(["LLM", sum(p.numel() for p in model.get_model().layers.parameters())])
        #         if hasattr(model.get_model(), "vision_tower"):
        #             table.add_row(["vision_tower", sum(p.numel() for p in model.get_model().vision_tower.parameters())])
        #             table.add_row(["vision_resampler", sum(p.numel() for p in model.get_model().vision_resampler.parameters())])
        #             table.add_row(["mm_projector", sum(p.numel() for p in model.get_model().mm_projector.parameters())])
        #         if hasattr(model.get_model(), "map_encoder"):
        #             table.add_row(["map_encoder", sum(p.numel() for p in model.get_model().map_encoder.parameters())])
        #             table.add_row(["map_resampler", sum(p.numel() for p in model.get_model().map_resampler.parameters())])
        #             table.add_row(["map_projector", sum(p.numel() for p in model.get_model().map_projector.parameters())])
        #         if hasattr(model.get_model(), "perception_encoder"):
        #             table.add_row(["perception_encoder", sum(p.numel() for p in model.get_model().perception_encoder.parameters())])
        #             table.add_row(["perception_resampler", sum(p.numel() for p in model.get_model().perception_resampler.parameters())])
        #             table.add_row(["perception_projector", sum(p.numel() for p in model.get_model().perception_projector.parameters())])
        #             table.add_row(["obj_projector", sum(p.numel() for p in model.get_model().obj_projector.parameters())])
        #         f.write(str(table))
        # with open(f'{training_args.output_dir}/{model_stat_path}/model_structure.txt', 'w') as f:
        #     f.write(str(model))
        with open(f'{training_args.output_dir}/{model_stat_path}/all_args.json', 'w') as f:
            args = {
                "model_args": model_args.__dict__,
                "data_args": data_args.__dict__,
                "training_args": training_args.__dict__,
                "model.config": model.config.__dict__,
            }
            def custom_default(obj):
                if type(obj) not in [str, int, float, bool, dict, list, tuple]:
                    return type(model).__name__
            f.write(json.dumps(args, indent=4, default=custom_default))
        print(f"Model and args saved to {training_args.output_dir}")


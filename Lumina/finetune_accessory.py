# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-Image-Accessory using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
import warnings
import torch.nn.functional as F

from PIL import Image, ImageFilter
# import cairosvg
from diffusers import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import cv2
from transformers import pipeline
import prodigyopt

from data import DataNoReportException, ItemProcessor, MyDataset
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
import models_accessory as models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue
from util.mask_tools import generate_random_mask
from models_accessory.lora import replace_linear_with_lora


#############################################################################
#                            Condition Generator                            #
#############################################################################
# Global variable to store loaded models
depth_pipeline = None

def get_canny_edges(image):
    canny_edge = cv2.Canny(np.array(image)[:, :, ::-1].copy(), 100, 200)
    canny_edge = torch.tensor(canny_edge)/255.0
    canny_edge = Image.fromarray((canny_edge.squeeze().numpy() * 255).astype(np.uint8))
    return canny_edge

def get_depth_map(image):
    global depth_pipeline
    if depth_pipeline is None:
        depth_pipeline = pipeline(
                            task="depth-estimation",
                            model="LiheYoung/depth-anything-small-hf",
                            device="cuda",
        )
    source_image = image.convert("RGB")
    depth_map = depth_pipeline(source_image)["depth"].convert("RGB")
    return depth_map

def get_masked_image(image):
    mask_types = ["left", "right", "top", "bottom", "center", "surrounding", "random_shape", "random_stroke", "all"]
    sample_prob = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    w, h = image.size
    mask_type = random.choices(mask_types, weights=sample_prob)[0]
    fill_mask = generate_random_mask(w, h, mask_type)
    white_bg = Image.new('RGB', (w, h), (255, 255, 255))
    mask_pil = Image.fromarray(((1 - fill_mask).squeeze().numpy() * 255).astype(np.uint8)).convert('L')
    masked_image = Image.composite(image, white_bg, mask_pil)
    return masked_image

def get_brightness_map(image):
    image = image.filter(ImageFilter.GaussianBlur(6))
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lower_thresh = 85  # Lower threshold
    upper_thresh = 170 # Higher threshold
    brightness_map = np.zeros_like(gray_image)
    brightness_map[gray_image <= lower_thresh] = 0  # Black
    brightness_map[(gray_image > lower_thresh) & (gray_image <= upper_thresh)] = 128  # Gray 
    brightness_map[gray_image > upper_thresh] = 255  # White
    brightness_map = Image.fromarray(brightness_map.astype(np.uint8))
    return brightness_map

def get_palette_map(image, num_colors=8):
    """Convert image to color blocks with specified number of colors"""
    image = image.filter(ImageFilter.GaussianBlur(12))
    w, h = image.size
    # First resize image to reduce details
    small_img = image.resize((w // 32, h // 32), Image.Resampling.NEAREST)
    
    # Convert to numpy array for processing
    img_array = np.array(small_img)
    pixels = img_array.reshape(-1, 3)
    
    # Use K-means clustering to find main colors
    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Replace each pixel with its nearest cluster center color
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    
    # Convert back to PIL image and resize to original size
    palette_map = Image.fromarray(quantized)
    palette_map = palette_map.resize((w, h), Image.Resampling.NEAREST)  # Use NEAREST to keep color block boundaries clear
    
    return palette_map

#############################################################################
#                            Data item Processor                            #
#############################################################################

def resize_with_aspect_ratio(img, resolution=1024, divisible=64, aspect_ratio=None):
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        w, h = img.size
        
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    # Ensure divisible
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    if is_tensor:
        mode = 'bilinear'
        align_corners = False
        if batch_dim:
            return F.interpolate(img, size=(new_h, new_w), 
                               mode=mode, align_corners=align_corners)
        else:
            return F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                               mode=mode, align_corners=align_corners).squeeze(0)
    else:
        return img.resize((new_w, new_h), Image.LANCZOS)

class NonRGBError(DataNoReportException):
    pass

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform
        self.special_format_set = set()
        self.cond_resolution = 1024
        self.img_resolution = 1024

    def text_image_pair_processor(self, data_item, task_type=None):
        task_types = ["t2i", "infilling", "canny", "depth", "brightness", "palette"]
        sample_prob = [1, 1, 1, 1, 1, 1]
        if task_type is None:
            task_type = random.choices(task_types, weights=sample_prob)[0]
        url = data_item["image_path"]
        try:
            image = Image.open(url)
            text = data_item["prompt"]
        except Exception as e:
            image = Image.new('RGB', (1024, 1024), color='black')
            text = "a black image"
            print(e)
            print("CONNOT OPEN", url)
        image = resize_with_aspect_ratio(image, self.img_resolution)

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            try:
                image = image.convert("RGB")
            except Exception as e:
                raise NonRGBError()

        if task_type == "t2i":
            cond_images = []
            system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "
        elif task_type == "infilling":
            cond_images = [get_masked_image(image)]
            system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a partially masked image. <Prompt Start> "
        elif task_type == "canny":
            cond_images = [get_canny_edges(image)]   
            system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a canny edge condition. <Prompt Start> "
        elif task_type == "depth":
            cond_images = [get_depth_map(image)]
            system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a depth map condition. <Prompt Start> "
        elif task_type == "brightness":
            cond_images = [get_brightness_map(image)]
            system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a brightness map condition. <Prompt Start> "
        elif task_type == "palette":
            cond_images = [get_palette_map(image)]
            system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a palette map condition. <Prompt Start> "
        
        for i in range(len(cond_images)):
            if isinstance(cond_images[i], Image.Image) and cond_images[i].mode != 'RGB':
                cond_images[i] = cond_images[i].convert('RGB')

        image = self.image_transform(image)
        cond_images = [self.image_transform(cond_image) for cond_image in cond_images]
        text = system_prompt + text
        cond_images = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond_images]
        return image, text, cond_images

    def subject_driven_processor(self, data_item):
        image = Image.open(data_item["input_image"])
        cond = [Image.open(data_item["output_image"])]
        prompt = data_item["prompt"]
        text = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and an object reference. <Prompt Start> " + prompt
        image = self.image_transform(image)
        cond = [self.image_transform(cond_image) for cond_image in cond]
        image = resize_with_aspect_ratio(image, self.img_resolution)
        cond = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond]
        return image, text, cond

    def editing_processor(self, data_item):
        image = Image.open(data_item["input_image"])
        cond = [Image.open(data_item["output_image"])]
        prompt = data_item["prompt"]
        text = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on editing instructions and an original image. <Prompt Start> " + prompt
        image = self.image_transform(image)
        cond = [self.image_transform(cond_image) for cond_image in cond]
        image = resize_with_aspect_ratio(image, self.img_resolution)
        cond = [resize_with_aspect_ratio(cond_image, self.cond_resolution) for cond_image in cond]
        return image, text, cond

    def process_item(self, data_item, group_name=None, training_mode=False):
        return data_item, group_name


#############################################################################
#                           Training Helper Functions                       #
#############################################################################

def apply_average_pool(latent, factor):
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)

def dataloader_collate_fn(samples):
    data_items = [x[0] for x in samples]
    group_names = [x[1] for x in samples]
    return data_items, group_names

def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    # 每个进程需要处理的样本总数
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    # 分别用于跟踪当前的epoch编号、已填充的样本索引数量以及偏移量，初始值均为0。
    epoch_id, fill_ptr, offs = 0, 0, 0
    # 直到fill_ptr达到sample_indices的大小，即所有需要的样本索引都已生成。
    while fill_ptr < sample_indices.size(0):
        # 新的随机数生成器g,打乱顺序
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        # 生成一个随机排列的索引数组epoch_sample_indices，该数组的长度等于数据集的大小
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        # 根据rank和offs计算出当前进程在这个epoch中应该处理的样本索引，并通过world_size进行步长切片，确保每个进程处理的数据不重叠。
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        # 更新offs以处理数据集大小不是world_size整数倍的情况，确保所有样本都能被均匀分配。
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        # 将计算出的索引epoch_sample_indices填充到sample_indices中。
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        # 使用tokenizer对captions列表中的每个提示进行编码。编码时，启用了填充（padding）、截断（truncation）和将长度填充到8的倍数（pad_to_multiple_of=8）的选项，
        # 并且最大长度限制为256。编码结果是一个包含input_ids和attention_mask的字典，分别表示编码后的文本ID和注意力掩码。
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask  # 用于指示哪些位置是填充的，哪些位置是实际的文本内容。

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        # 这里指定了output_hidden_states=True，以便获取模型的所有隐藏状态，然后从隐藏状态中取出倒数第二层作为最终的文本嵌入（prompt_embeds）。

    return prompt_embeds, prompt_masks


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    print(f"Starting rank={rank}, seed={seed}, "
          f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
            )
        )
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")

    # create tokenizers
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"

    # create text encoders
    # text_encoder
    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
    ).cuda()
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    logger.info(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size

    # Create model:
    model = models.__dict__[args.model](
        in_channels=16,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size
    
    # If using LoRA, apply LoRA to the model
    if args.training_type == "lora":
        logger.info(f"Applying LoRA with rank {args.lora_rank} and scale {args.lora_scale}")
        replace_linear_with_lora(model, max_rank=args.lora_rank, scale=args.lora_scale)
    
    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                latest_ckpt = os.path.join(checkpoint_dir, 'latest.pth')
                if os.path.exists(latest_ckpt):
                    args.resume = latest_ckpt
                else:
                    existing_checkpoints.sort()
                    args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
            model.init_cond_refiner()
            model_ema.init_cond_refiner()   
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        logger.info("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"model:\n{model}\n")
    
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(
        device
    )

    logger.info("AdamW eps 1e-15 betas (0.9, 0.95)")

    model_params = []
    for name, param in model.named_parameters():
        if args.training_type == "full_model":
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "bias" and 'bias' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "norm" and 'norm' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "lora" and 'lora' in name:
            param.requires_grad = True
            model_params.append(param)
        elif args.training_type == "attention" and ('attention' in name or 'cond' in name):
            param.requires_grad = True
            model_params.append(param)
        else:
            param.requires_grad = False

    # opt = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    opt = prodigyopt.Prodigy(model_params, lr=args.lr, use_bias_correction=True, safeguard_warmup=True, weight_decay=args.wd, slice_p=11)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    data_collection = {}
    for train_res in [1024]:
        logger.info(f"Creating data for resolution {train_res}")

        global_bsz = getattr(args, f"global_bsz_{train_res}")
        local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
        micro_bsz = getattr(args, f"micro_bsz_{train_res}")
        assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
        logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")

        patch_size = 8 * model_patch_size
        logger.info(f"patch size: {patch_size}")
        max_num_patches = round((train_res / patch_size) ** 2)
        logger.info(f"Limiting number of patches to {max_num_patches}.")
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
        logger.info("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
        image_transform = transforms.Compose(
            [
                # transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        t2i_item_processor = T2IItemProcessor(image_transform)
        dataset = MyDataset(
            args.data_path,
            item_processor=t2i_item_processor,
            cache_on_disk=args.cache_data_on_disk,
        )
        num_samples = global_bsz * args.max_steps
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
        sampler = get_train_sampler(
            dataset,
            dp_rank,
            dp_world_size,
            global_bsz,
            args.max_steps,
            resume_step,
            args.global_seed + train_res * 100,  # avoid same sampling for different resolutions
        )
        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )

        # default: 1000 steps, linear noise schedule
        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=args.snr_type,
            do_shift=not args.no_shift,
            seq_len=(train_res // 16) ** 2,
        )  # default: velocity;

        data_collection[train_res] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
            "transport": transport,
        }

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:

    logger.info(f"Training for {args.max_steps:,} steps...")

    for step in range(resume_step, args.max_steps):
        start_time = time()
        for train_res, data_pack in data_collection.items():
            # image process after loading
            data_items, group_names = next(data_pack["loader_iter"])
            x, caps, cond, position_type = [], [], [], []
            for data_item, group_name in zip(data_items, group_names):
                if group_name == "text_image_pair":
                    # inpaint image
                    x_, caps_, cond_ = t2i_item_processor.text_image_pair_processor(data_item)
                    x.append(x_)
                    caps.append(caps_)
                    cond.append(cond_)
                    position_type.append(["aligned"])
                elif group_name == "subject_driven":
                    x_, caps_, cond_ = t2i_item_processor.subject_driven_processor(data_item)
                    x.append(x_)
                    caps.append(caps_)
                    cond.append(cond_)
                    position_type.append(["offset"])
                elif group_name == "editing":
                    x_, caps_, cond_ = t2i_item_processor.editing_processor(data_item)
                    x.append(x_)
                    caps.append(caps_)
                    cond.append(cond_)
                    position_type.append(["aligned"])
                else:
                    raise ValueError(f"Unrecognized item: {data_item}")


            x = [img.to(device, non_blocking=True) for img in x]
            cond = [[cond_img.to(device, non_blocking=True) for cond_img in cond_imgs] for cond_imgs in cond]

            with torch.no_grad():
                vae_scale = {
                    "sdxl": 0.13025,
                    "sd3": 1.5305,
                    "ema": 0.18215,
                    "mse": 0.18215,
                    "cogvideox": 1.15258426,
                    "flux": 0.3611,
                }["flux"]
                vae_shift = {
                    "sdxl": 0.0,
                    "sd3": 0.0609,
                    "ema": 0.0,
                    "mse": 0.0,
                    "cogvideox": 0.0,
                    "flux": 0.1159,
                }["flux"]

                if step == resume_step:
                    warnings.warn(f"vae scale: {vae_scale}    vae shift: {vae_shift}")
                # Map input images to latent space + normalize latents:

                for i, img in enumerate(x):
                    x[i] = (vae.encode(img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale
                    x[i] = x[i].float()
                    cond[i] = [(vae.encode(cond_img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale for cond_img in cond[i]]
                    cond[i] = [cond_img.float() for cond_img in cond[i]]

            with torch.no_grad():
                cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

            loss_item = 0.0
            loss_256_item = 0.0
            loss_1024_item = 0.0

            opt.zero_grad()

            # Number of bins, for loss recording
            n_loss_bins = 20
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            loss_bins_256 = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
            bin_occurrence_256 = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss_256 = torch.zeros(n_loss_bins, device="cuda")

            for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
                mb_st = mb_idx * data_pack["micro_bsz"]
                mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
                last_mb = mb_ed == data_pack["local_bsz"]

                x_mb = x[mb_st:mb_ed]
                cond_mb = cond[mb_st:mb_ed]
                position_type_mb = position_type[mb_st:mb_ed]
                ### muti resolution
                x_mb_256 = [apply_average_pool(x, 2) for x in x_mb]
                cond_mb_256 = [[apply_average_pool(cond_img, 2) for cond_img in cond_imgs] for cond_imgs in cond_mb]

                cap_feats_mb = cap_feats[mb_st:mb_ed]
                cap_mask_mb = cap_mask[mb_st:mb_ed]

                model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb, cond=cond_mb, position_type=position_type_mb)
                model_kwargs_256 = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb, cond=cond_mb_256, position_type=position_type_mb)

                with {
                    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[args.precision]:
                    loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs)
                    loss_dict_256 = data_pack["transport"].training_losses(model, x_mb_256, model_kwargs_256)
                loss_1024 = loss_dict["loss"].sum() / data_pack["local_bsz"]
                loss_256 = loss_dict_256["loss"].sum() / data_pack["local_bsz"]
                loss_256 = torch.zeros(1).to(loss_1024.dtype).to(loss_1024.device)
                loss = loss_1024 + loss_256
                loss_item += loss.item()
                loss_1024_item += loss_1024.item()
                loss_256_item += loss_256.item()
                with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                    loss.backward()

                # for bin-wise loss recording
                # Digitize t values to find which bin they belong to
                bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                detached_loss = loss_dict["loss"].detach()
                bin_indices_256 = torch.bucketize(loss_dict_256["t"].cuda(), loss_bins, right=True) - 1
                detached_loss_256 = loss_dict_256["loss"].detach()

                # Iterate through each bin index to update occurrence and sum
                for i in range(n_loss_bins):
                    mask = bin_indices == i  # Mask for elements in the i-th bin
                    bin_occurrence[i] = bin_occurrence[i] + mask.sum()  # Count occurrences in the i-th bin
                    bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()  # Sum loss values in the i-th bin
                for i in range(n_loss_bins):
                    mask = bin_indices_256 == i
                    bin_occurrence_256[i] = bin_occurrence_256[i] + mask.sum()
                    bin_sum_loss_256[i] = bin_sum_loss_256[i] + detached_loss_256[mask].sum()

            grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)

            dist.all_reduce(bin_occurrence)
            dist.all_reduce(bin_sum_loss)
            dist.all_reduce(bin_occurrence_256)
            dist.all_reduce(bin_sum_loss_256)
            if tb_logger is not None:
                tb_logger.add_scalar(f"{train_res}/loss", loss_item, step)
                tb_logger.add_scalar(f"{train_res}/loss_256", loss_256_item, step)
                tb_logger.add_scalar(f"{train_res}/loss_1024", loss_1024_item, step)
                tb_logger.add_scalar(f"{train_res}/grad_norm", grad_norm, step)
                tb_logger.add_scalar(f"{train_res}/lr", opt.param_groups[0]["lr"], step)
                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        tb_logger.add_scalar(f"{train_res}/loss-bin{i+1}-{n_loss_bins}", bin_avg_loss, step)
                for i in range(n_loss_bins):
                    if bin_occurrence_256[i] > 0:
                        bin_avg_loss = (bin_sum_loss_256[i] / bin_occurrence_256[i]).item()
                        tb_logger.add_scalar(f"{train_res}/loss_256-bin{i+1}-{n_loss_bins}", bin_avg_loss, step)

            opt.step()
            end_time = time()

            # Log loss values:
            metrics = data_pack["metrics"]
            metrics["loss"].update(loss_item)
            metrics["loss_1024"].update(loss_1024_item)
            metrics["loss_256"].update(loss_256_item)
            metrics["grad_norm"].update(grad_norm)
            metrics["Secs/Step"].update(end_time - start_time)
            metrics["Imgs/Sec"].update(data_pack["global_bsz"] / (end_time - start_time))
            metrics["grad_norm"].update(grad_norm)
            for i in range(n_loss_bins):
                if bin_occurrence[i] > 0:
                    bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                    metrics[f"bin_1024_{i + 1:02}-{n_loss_bins}"].update(bin_avg_loss, int(bin_occurrence[i].item()))
            for i in range(n_loss_bins):
                if bin_occurrence_256[i] > 0:
                    bin_avg_loss = (bin_sum_loss_256[i] / bin_occurrence_256[i]).item()
                    metrics[f"bin_256_{i + 1:02}-{n_loss_bins}"].update(bin_avg_loss, int(bin_occurrence_256[i].item()))
            if (step + 1) % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                logger.info(
                    f"Res{train_res}_{train_res//4}: (step{step + 1:07d}) "
                    + f"lr{opt.param_groups[0]['lr']:.6f} "
                    + " ".join([f"{key}:{str(metrics[key])}" for key in sorted(metrics.keys())])
                )

            start_time = time()

        update_ema(model_ema, model)

        # Save DiT checkpoint:
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps or (step + 1) % 100 == 0:
            if (step + 1) % args.ckpt_every == 0:
                checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            else:
                checkpoint_path = f"{checkpoint_dir}/latest"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2_Adaln_Refiner")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_bsz_1024", type=int, default=256)
    parser.add_argument("--micro_bsz_1024", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--training_type", type=str, default="full_model")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument(
        "--no_shift",
        action="store_true",
    )
    # 添加LoRA相关参数
    parser.add_argument("--lora_rank", type=int, default=128, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for LoRA adaptation")
    args = parser.parse_args()

    main(args)

import argparse
import json
import math
import os
import random
import socket
import time

from diffusers.models import AutoencoderKL
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import cv2
from transformers import pipeline

import torch.nn.functional as F

from data import DataNoReportException, ItemProcessor, MyDataset
from torchvision import transforms

import models_accessory as models
from transport import Sampler, create_transport
from models_accessory.lora import replace_linear_with_lora
import gradio as gr

#############################################################################
#                            Condition Generator                            #
#############################################################################
# Global variables to store loaded models
depth_pipeline = None
model = None
vae = None
text_encoder = None
tokenizer = None
sampler = None

# System prompt options
# SYSTEM_PROMPTS = {
#     "Text-to-Image": "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> ",
#     "Image Infilling": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a partially masked image. <Prompt Start> ",
#     "Image Restoration": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a degraded image. <Prompt Start> ",
#     "Edge Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a canny edge condition. <Prompt Start> ",
#     "Depth Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a depth map condition. <Prompt Start> ",
#     "Brightness Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a brightness map condition. <Prompt Start> ",
#     "Palette Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a palette map condition. <Prompt Start> ",
#     "Human Keypoint Condition": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a human keypoint condition. <Prompt Start> ",
#     "Subject-driven": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and an object reference. <Prompt Start> ",
#     "Image Relighting": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on image-lighting instructions and an original image. <Prompt Start> ",
#     "Image Editing": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on editing instructions and an original image. <Prompt Start> ",
# }

SYSTEM_PROMPTS = {
    "Image Infilling": "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a partially masked image. <Prompt Start> ",
}

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
                            torch_dtype=torch.float32
        )
    source_image = image.convert("RGB")
    with torch.cuda.amp.autocast(enabled=False):
        depth_output = depth_pipeline(source_image)
    return depth_output["depth"].convert("RGB")

def get_brightness_map(image):
    image = image.filter(ImageFilter.GaussianBlur(6))
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lower_thresh = 85  
    upper_thresh = 170 
    brightness_map = np.zeros_like(gray_image)
    brightness_map[gray_image <= lower_thresh] = 0 
    brightness_map[(gray_image > lower_thresh) & (gray_image <= upper_thresh)] = 128 
    brightness_map[gray_image > upper_thresh] = 255 
    brightness_map = Image.fromarray(brightness_map.astype(np.uint8))
    return brightness_map

def get_palette_map(image, num_colors=8):
    image = image.filter(ImageFilter.GaussianBlur(12))
    w, h = image.size
    small_img = image.resize((w // 32, h // 32), Image.Resampling.NEAREST)
    
    img_array = np.array(small_img)
    pixels = img_array.reshape(-1, 3)
    
    pixels = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    
    palette_map = Image.fromarray(quantized)
    palette_map = palette_map.resize((w, h), Image.Resampling.NEAREST) 
    
    return palette_map
def get_masked_image(image_edit):
    image = image_edit['composite']
    # Convert the edited image to a binary mask
    # Assuming the drawn areas are marked in a specific color (e.g., black or any non-white color)
    import numpy as np
    from PIL import Image
    
    # Convert image to grayscale
    gray_image = image.convert("L")
    # Convert to numpy array
    np_image = np.array(gray_image)
    
    # Create a binary mask: set drawn areas to 255, others to 0
    mask = np.where(np_image < 255, 255, 0).astype(np.uint8)
    # Convert back to PIL Image
    mask_image = Image.fromarray(mask)
    return mask_image


def process_image_mask(editor_state):
    # 获取合成后的图像
    composite_image = editor_state['composite']
    
    # 将图像转换为 numpy 数组
    img_array = np.array(composite_image)
    print(img_array.shape)
    # 创建一个全白的图像
    white_image = np.ones_like(img_array) * 255
    
    # 假设用户使用红色画笔涂抹
    # 创建红色通道的掩码（红色值高，绿色和蓝色值低）
    red_threshold = 200
    green_blue_threshold = 100
    
    # 创建掩码：红色通道 > red_threshold 且 绿色和蓝色通道 < green_blue_threshold
    mask = (img_array[:, :, 0] > red_threshold) & \
           (img_array[:, :, 1] < green_blue_threshold) & \
           (img_array[:, :, 2] < green_blue_threshold)
    
    # 如果是 RGBA 图像，保留原始 alpha 通道
    if img_array.shape[-1] == 4:
        white_image[:, :, 3] = img_array[:, :, 3]
    
    # 应用掩码：将涂抹区域设为白色
    result = img_array.copy()
    result[mask] = white_image[mask]
    
    # 将图像转换为 RGB 模式
    if result.shape[-1] == 4:  # 如果有 alpha 通道
        # 创建一个白色背景的 RGB 图像
        background = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
        # 将 alpha 通道应用到图像上
        alpha = result[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        result_rgb = (result[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    else:
        result_rgb = result
    
    return Image.fromarray(result_rgb)



def process_image_mask_v2(paint) :
    image, mask = paint['background'], paint['layers'][0].convert('L')
    image, mask = np.array(image), np.array(mask) # mask: wxh, 画笔涂抹区域为白色,其余为原来正常颜色

    # 将mask中大于128的像素点转换为255,直接赋值给image
    mask_bool = mask > 128
    image[mask_bool] = 255

    # mask_bool = np.where(mask > 128, True, False)
    # image[mask_bool] = image[mask_bool, ::-1] # RGB->BGR, 可能用于图像编辑工具中，允许用户在特定区域修改颜色通道顺序（例如模拟 BGR 格式的视觉效果）。
    return Image.fromarray(image)



#############################################################################
#                            Data item Processor                            #
#############################################################################

def resize_with_aspect_ratio(img, resolution=1024, divisible=64, aspect_ratio=None):
    """resize the image with aspect ratio, keep the area close to resolution**2, and the width and height can be divisible by divisible"""
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        if img.dim() == 3:
            c, h, w = img.shape
            batch_dim = False
        else:
            b, c, h, w = img.shape
            batch_dim = True
    else:
        print("img.size:", img)
        w, h = img.size
        
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
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

def encode_prompt(prompt_batch, text_encoder, tokenizer):
    """encode the text prompt"""
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

def load_models(args):
    print("Loading models...")
    global model, vae, text_encoder, tokenizer, sampler
    
    # Set data type
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_path="/root/paddlejob/workspace/env_run/output/lvfeng/model/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(text_path)
    tokenizer.padding_side = "right"
    print("Tokenizer loaded")
    
    text_encoder = AutoModel.from_pretrained(
        text_path, torch_dtype=dtype, device_map="cuda"
    ).eval()
    print("Text encoder loaded")
    cap_feat_dim = text_encoder.config.hidden_size
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("/root/paddlejob/workspace/env_run/output/lvfeng/model/vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    print("VAE loaded")
    # Create model
    model = models.__dict__["NextDiT_2B_GQA_patch2_Adaln_Refiner"](
        in_channels=16,
        qk_norm=True,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to(device, dtype=dtype)
    print("Model created")
    # Load model weights
    # ckpt_path="/root/paddlejob/workspace/env_run/output/lvfeng/model/models--Alpha-VLLM--Lumina-Accessory/snapshots/711d5d6656c62957e8625b02ea53cc74f2c5589d/consolidated.00-of-01.pth"
    replace_linear_with_lora(model, max_rank = 128, scale = 1.0)
    # lora
    # ckpt_path="model/plain_model.pt"
    ckpt_path = "lumina_results/checkpoints/0010000/consolidated.00-of-01.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    print("Model weights loaded")   
    # Create sampler
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    
    print("All models loaded successfully")

def process_image(image, resolution=1024):
    if image is None:
        return None
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    return transform(image)

def generate_image(cond_image, prompt, system_prompt, num_steps=50, cfg_scale=4.0, t_shift=6, seed=None, resolution=1024):
    """Main function for image generation"""
    global model, vae, text_encoder, tokenizer, sampler
    
    if model is None:
        return None, "Models not loaded yet, please check model path"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # try:
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Process condition image
        if cond_image is not None:
            cond_image = resize_with_aspect_ratio(cond_image, resolution)
            cond_tensor = process_image(cond_image)
            cond_tensor = cond_tensor.to(device)
            
            # VAE encoding
            vae_scale = 0.3611
            vae_shift = 0.1159
            cond_latent = (vae.encode(cond_tensor[None].to(dtype)).latent_dist.mode()[0] - vae_shift) * vae_scale
            cond_latent = cond_latent.float()
            cond = [[cond_latent]]
            if "object reference" in system_prompt:
                position_type = [["offset"]]
            else:
                position_type = [["aligned"]]
        else:
            raise ValueError("Condition image cannot be empty")
        
        # Full prompt
        full_prompt = system_prompt + prompt
        print(full_prompt)
        
        # Encode prompt
        cap_feats, cap_mask = encode_prompt([full_prompt] + [""], text_encoder, tokenizer)
        cap_mask = cap_mask.to(cap_feats.device)
        
        # Prepare model parameters
        w, h = cond_latent.shape[-2:] if cond_image is not None else (128, 128)
        z = torch.randn([1, 16, w, h], device=device).to(dtype)
        z = z.repeat(2, 1, 1, 1)
        cond = cond * 2
        position_type = position_type * 2

        model_kwargs = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            cfg_scale=cfg_scale,
            cond=cond,
            position_type=position_type,
        )
        
        # Sampling
        sample_fn = sampler.sample_ode(
                            sampling_method=args.solver,
                            num_steps=num_steps,
                            atol=args.atol,
                            rtol=args.rtol,
                            reverse=args.reverse,
                            time_shifting_factor=t_shift
                        )
        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
        
        # Decode generated image
        samples = vae.decode(samples / vae.config.scaling_factor + vae.config.shift_factor)[0]
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)
        
        # Convert to PIL image
        output_image = to_pil_image(samples[0].float().cpu())
        
        return output_image, "Successfully generated image! \n System prompt: " + system_prompt + "\n Prompt: " + prompt
    
    # except Exception as e:
    #     return None, f"Error during generation: {str(e)}"

def create_demo():
    """Create Gradio demo interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Traffic T2I/EDIT Demo")
        
        with gr.Row():
            with gr.Column():
                cond_image = gr.ImageEditor(label='Input', type='pil', image_mode='RGB', layers=False,
                                   brush=gr.Brush(colors=["#AAAAAA"], color_mode="fixed")) # 方法一:使用十六进制颜色代码,代表一种灰色,画笔的颜色是固定的
                # cond_image = gr.ImageEditor(label="Condition Image", type="pil", format="jpeg") # 方法二:存在RGBA4个通道
                prompt = gr.Textbox(label="Text Prompt", lines=3, placeholder="Enter text describing the image...") # 用于描述图像的文本
                system_prompt_dropdown = gr.Dropdown(
                    choices=list(SYSTEM_PROMPTS.keys()),
                    value="Image Infilling",
                    label="System Prompt Type"
                ) # 用于选择系统提示的类型（如“Text-to-Image”）。
                
                generate_btn1 = gr.Button("Generate Condition Image")
                with gr.Row():
                    resolution = gr.Slider(minimum=256, maximum=1024, value=1024, step=256, label="Inference Resolution")
                
                with gr.Row():
                    num_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Inference Steps")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    t_shift = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Time Shift Factor")
                    seed = gr.Number(label="Random Seed (leave empty for random)", precision=0, value=20)
                
                generate_btn = gr.Button("Generate Image")
                
            with gr.Column():
                with gr.Row():
                    condition_preview = gr.Image(label="Condition Preview", type='pil',  image_mode='RGB', format="jpg", interactive=False)
                output_image = gr.Image(label="Generated Image", format="jpg")
                output_message = gr.Textbox(label="Status Message")
                
                
            generate_btn1.click(
                fn=process_image_mask_v2,
                inputs=[cond_image],
                outputs=[condition_preview]
            )
        # set event handlers
        def get_system_prompt(choice):
            return SYSTEM_PROMPTS[choice]
            
        
        # generate image - first define this function
        def process_and_generate(img, txt, sys_prompt, steps, scale, t_shift_val, seed_val, resolution):
            if img is None:
                return None, None, "Please upload a condition image first"
            
            
            system_prompt = get_system_prompt(sys_prompt)
            output_img, message = generate_image(
                img, txt, system_prompt, steps, scale, t_shift_val,
                int(seed_val) if seed_val else None, resolution
            )
            
            return output_img, message
        
        generate_btn.click(
            fn=process_and_generate,
            inputs=[condition_preview, prompt, system_prompt_dropdown,
                   num_steps, cfg_scale, t_shift, seed, resolution],
            outputs=[output_image, output_message]
        )
    return demo

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )



def none_or_str(value):
    if value == "None":
        return None
    return value


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    print("Parsing command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None) # required=True
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--t_shift", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.jpeg in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="None",
        choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--system_type",
        type=str,
        default="real",
        # choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse", "sdxl", "flux"], default="flux"
    )
    parser.add_argument(
        "--text_encoder", type=str, nargs='+', default=['gemma'], help="List of text encoders to use (e.g., t5, clip, gemma)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max length for text encoder."
    )
    parser.add_argument(
        "--use_parallel_attn",
        type=bool,
        default=False,
        help="Use parallel attention in the model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        type=bool,
        default=True,
        help="Use Flash Attention in the model.",
    )
    parser.add_argument("--do_shift", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--training_type", type=str, default="full_model")
    parser.add_argument("--lora_rank", type=int, default=128, help="Rank for LoRA adaptation")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for LoRA adaptation")

    parse_transport_args(parser)
    parse_ode_args(parser)

    parser.add_argument("--share", action="store_true", help="是否共享Gradio演示")

    args = parser.parse_known_args()[0]
    print(f"Arguments parsed: {args}")
    
    # Load models first
    load_models(args)
    
    print("Preparing to launch Gradio demo")
    demo = create_demo()
    print("Gradio demo created, preparing to launch")
    demo.queue().launch(
        server_name="10.96.203.76",
        server_port=8010,
        share=False,
        max_threads=10
    )
    print("Gradio demo launched successfully")

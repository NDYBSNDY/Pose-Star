import os
import argparse
import gradio as gr
from datetime import datetime

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.flux.pipeline_flux_tryon import FluxTryOnPipeline
from utils import resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX Try-On Demo")
    parser.add_argument(
        "--base_model_path",
        type=str,
        # default="black-forest-labs/FLUX.1-Fill-dev",
        default="Models/FLUX.1-Fill-dev",
        help="The path to the base model to use for evaluation."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="The Path to the checkpoint of trained tryon model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Whether or not to allow TF32 on Ampere GPUs."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="The width of the input image."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of the input image."
    )
    return parser.parse_args()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def submit_function_flux(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):

    # Process image editor input
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    # Set random seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    # Process input images
    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    
    # Adjust image sizes
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    result_image = pipeline_flux(
        image=person_image,
        condition_image=cloth_image,
        mask_image=mask,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    # Post-processing
    masked_person = vis_mask(person_image, mask)

    # Return result based on show type
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
        
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
        return new_result_image

def person_example_fn(image_path):
    return image_path


def app_gradio():
    with gr.Blocks(title="CatVTON with FLUX.1-Fill-dev") as demo:
        gr.Markdown("# CatVTON with FLUX.1-Fill-dev")
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    image_path_flux = gr.Image(
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                    person_image_flux = gr.ImageEditor(
                        interactive=True, label="Person Image", type="filepath"
                    )
                
                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image_flux = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">Two ways to provide Mask:<br>1. Upload the person image and use the `🖌️` above to draw the Mask (higher priority)<br>2. Select the `Try-On Cloth Type` to generate automatically </span>'
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "lower", "overall"],
                            value="upper",
                        )

                submit_flux = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps_flux = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
                    # Guidence Scale
                    guidance_scale_flux = gr.Slider(
                        label="CFG Strenth", minimum=0.0, maximum=50, step=0.5, value=30
                    )
                    # Random Seed
                    seed_flux = gr.Slider(
                        label="Seed", minimum=-1, maximum=10000, step=1, value=42
                    )
                    show_type = gr.Radio(
                        label="Show Type",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                    )
                
            with gr.Column(scale=2, min_width=500):
                result_image_flux = gr.Image(interactive=False, label="Result")
                with gr.Row():
                    # Photo Examples
                    root_path = "resource/demo/example"
                    with gr.Column():
                        gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "men", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ],
                            examples_per_page=4,
                            inputs=image_path_flux,
                            label="Person Examples ①",
                        )
                        gr.Examples(
                            examples=[
                                os.path.join(root_path, "person", "women", _)
                                for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ],
                            examples_per_page=4,
                            inputs=image_path_flux,
                            label="Person Examples ②",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Person examples come from the demos of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a>. </span>'
                        )
                    with gr.Column():
                        gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "upper", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image_flux,
                            label="Condition Upper Examples",
                        )
                        gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "overall", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image_flux,
                            label="Condition Overall Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=[
                                os.path.join(root_path, "condition", "person", _)
                                for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                            ],
                            examples_per_page=4,
                            inputs=cloth_image_flux,
                            label="Condition Reference Person Examples",
                        )
                        gr.Markdown(
                            '<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>'
                        )

                
            image_path_flux.change(
                person_example_fn, inputs=image_path_flux, outputs=person_image_flux
            )

            submit_flux.click(
                submit_function_flux,
                [person_image_flux, cloth_image_flux, cloth_type, num_inference_steps_flux, guidance_scale_flux, seed_flux, show_type],
                result_image_flux,
            )
        
    
    demo.queue().launch(share=True, show_error=True)

# 解析参数
args = parse_args()

# 加载模型
repo_path = snapshot_download(repo_id=args.resume_path)
pipeline_flux = FluxTryOnPipeline.from_pretrained(args.base_model_path)
pipeline_flux.load_lora_weights(
    os.path.join(repo_path, "flux-lora"), 
    weight_name='pytorch_lora_weights.safetensors'
)
pipeline_flux.to("cuda", torch.bfloat16)

# 初始化 AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8, 
    do_normalize=False, 
    do_binarize=True, 
    do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda'
)

if __name__ == "__main__":
    app_gradio()

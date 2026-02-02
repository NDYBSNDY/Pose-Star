import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import DDIMScheduler
from function_SD3 import invert, sample, null_optimization

# Useful function for later
def load_image(url, size=None):
    # response = requests.get(url, timeout=0.2)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    img = Image.open(url).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load a pipeline stable-diffusion-v1-5
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# stable-diffusion-3.5
# ---------------------------
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
model_id = "/data2/infer/stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    # bnb_4bit_compute_dtype=torch.float16
    
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
    # torch_dtype=torch.float16
    # torch_dtype=torch.float32
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
    # torch_dtype=torch.float16
    # torch_dtype=torch.float32
)
pipe.enable_model_cpu_offload()
# ---------------------------

# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# https://www.pexels.com/photo/a-beagle-on-green-grass-field-8306128/
# input_image = load_image("https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg", size=(512, 512))
path = "/data2/infer/res/Inversion/test/people.jpg"
# input_image = load_image(path, size=(512, 512))
# 输入图像尺寸需要为2的倍数，方便Dit分块
input_image = load_image(path, size=(512, 512))
# input_image_prompt = "Photograph of a puppy on the grass"
input_image_prompt = "Girls and boys"

# Encode with VAE
# 将此 PIL 图像转换为一组潜在值，我们将将其用作反转的起点
with torch.no_grad():
    a = tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
    a = a.to(dtype=torch.bfloat16)
    # latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
    latent = pipe.vae.encode(a)
l = 0.18215 * latent.latent_dist.sample()

inverted_latents = invert(l, input_image_prompt, pipe=pipe, num_inference_steps=50, do_classifier_free_guidance=True)
# null optimization
uncond_embeddings = null_optimization(pipe=pipe, prompt=input_image_prompt, ddim_latents=inverted_latents, num_inner_steps=10, early_stop_epsilon=1e-5, do_classifier_free_guidance=True)

# The reason we want to be able to specify start step
# start_step值越大与原图越相似，作弊程度越高
# 原始狗狗图像：6（伪影） - 7（约等于原图）的时候产生一个分界点
start_step = 47
img = sample(
    input_image_prompt,
    pipe=pipe,
    start_latents=inverted_latents[-(start_step + 1)][None],
    start_step=start_step,
    num_inference_steps=50,
    do_classifier_free_guidance=True,
    uncond_embeddings=uncond_embeddings
)[0]
img.save("/data2/infer/res/Inversion/res/people-1.jpg")














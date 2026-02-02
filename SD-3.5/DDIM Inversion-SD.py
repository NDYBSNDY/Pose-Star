import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import DDIMScheduler, StableDiffusionPipeline
from function_SD import invert, sample


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
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)


# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# https://www.pexels.com/photo/a-beagle-on-green-grass-field-8306128/
# input_image = load_image("https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg", size=(512, 512))
path = "/home/dongyuran/code/Inversion/test/women3.jpg"
# input_image = load_image(path, size=(512, 512))
input_image = load_image(path, size=(253, 142))
# input_image_prompt = "Photograph of a puppy on the grass"
input_image_prompt = " "

# Encode with VAE
with torch.no_grad():
    latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
l = 0.18215 * latent.latent_dist.sample()

# DDIM
inverted_latents = invert(l, input_image_prompt, pipe=pipe, num_inference_steps=50)

# The reason we want to be able to specify start step
# start_step值越大与原图越相似，作弊程度越高
# 原始狗狗图像：6（伪影） - 7（约等于原图）的时候产生一个分界点
start_step = 45
img = sample(
    input_image_prompt,
    pipe=pipe,
    start_latents=inverted_latents[-(start_step + 1)][None],
    start_step=start_step,
    num_inference_steps=50
)[0]
img.save("/home/dongyuran/code/Inversion/res/women3_1.jpg")
print("good")














import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "/data2/stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    # torch_dtype=torch.bfloat16
    torch_dtype=torch.float16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    # torch_dtype=torch.bfloat16
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

# prompt = "Aircraft flying to the left, civil aircraft, small aircraft"

# prompt = "Many zebras in a vast grassland, one of them with a pink bow on its head, a few clouds in the sky, and a few birds, and lions, mice, tigers, and rabbits in the grassland, and one of the rabbits is blue"

prompt = "A little girl in a sweater standing in the road"

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    # num_inference_steps=50,
    # guidance_scale=4.5,
    guidance_scale=6.5,
    max_sequence_length=512,
).images[0]
image.save("/data2/infer/res/SD-test/aircraft-18.png")

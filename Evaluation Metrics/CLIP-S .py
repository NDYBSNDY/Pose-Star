from torch import randint
from PIL import Image
import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore
import torchvision.transforms as transforms
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

path = "E:\\code\\Evaluation Metrics\\img\\generated - one\\4.png"

image = Image.open(path).convert("RGB")

transform = transforms.ToTensor()

tensor_img = transform(image)
score = metric(tensor_img, "A man wearing a hat")
score.detach().round()
print("score:")
print(score.item())
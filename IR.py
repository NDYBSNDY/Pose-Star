import ImageReward as RM
# model = RM.load("ImageReward-v1.0")
model = RM.load("/data2/dongyuran/code/ImageReward/ImageReward.pt")

rewards = model.score("A man in a black T-shirt", ["/data2/dongyuran/code/ImageReward/img/image (24).png", "/data2/dongyuran/code/ImageReward/img/image (25).png"])

print(rewards)

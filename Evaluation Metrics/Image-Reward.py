# pip install image-reward
import ImageReward as RM
model = RM.load("ImageReward-v1.0")

rewards = model.score("A woman with sunglasses on her head and a jacket on top", 
                      ["E:/code/Evaluation Metrics/img/generated/4.png", 
                       "E:/code/Evaluation Metrics/img/generated - p2p/4.png", ])
print(rewards)
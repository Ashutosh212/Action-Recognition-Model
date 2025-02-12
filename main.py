import torch
import os

from torchsummary import summary


model_path = os.path.join("MAE2", "vit_s_k710_dl_from_giant.pth")
# model = torch.load(model_path)
# # print(model)

# summary(model, input_size=(1, 10))

model = torch.load(model_path)
print(type(model))
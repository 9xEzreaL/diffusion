from PIL import Image
import os
import numpy as np
import torch

model = torch.load('/home/ziyi/Projects/OaiGan/OaiGan/submodels/model_seg256.pth')
model.eval()

path = '/media/ziyi/Dataset/TSE_DESS/TSE_DESS/test/b/'
identity = os.listdir(path)

os.makedirs(path.replace('/b/', '/b_mask/'), exist_ok=True)

for sub_identity in identity:
    img = Image.open(os.path.join(path, sub_identity))
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img - 0.5)/0.5
    img = img[19:275, 19:275]
    img = np.expand_dims(img, 0)

    img = np.concatenate([img, img, img], 0)
    img = torch.Tensor(np.expand_dims(img, 0 )).cuda()
    print(img.shape)
    mask = model(img)
    _, mask = torch.max(mask, 1)
    mask = np.squeeze(np.array(mask.detach().cpu()), 0).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(os.path.join(path.replace('/b/', '/b_mask/'), sub_identity))

# print(img.max())
# print(img.min())
# print(img.shape)
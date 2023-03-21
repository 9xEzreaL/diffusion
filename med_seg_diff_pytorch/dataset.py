import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
# import pandas as pd
import random
import albumentations as A
import torchvision.transforms.functional as F
import torch.nn.functional as fn



class ISICDataset(Dataset):
    def __init__(self, data_path, csv_file, img_folder, transform=None, training=True, flip_p=0.5):
        df = pd.read_csv(os.path.join(data_path, csv_file), encoding='gbk')
        self.img_folder = img_folder
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.flip_p = flip_p

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index] + '.jpg'
        img_path = os.path.join(self.data_path, self.img_folder, name)

        mask_name = name.split('.')[0] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, self.img_folder, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.training:
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
            if random.random() < self.flip_p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        if self.training:
            return (img, mask)
        return (img, mask, label)


class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        '''
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith('.npy')]

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        npy_img = np.load(os.path.join(self.directory, fname))
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)
        if self.test_flag:
            return image, mask, fname
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)

class OAIDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, mode: str='train'):
        '''
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        '''
        super().__init__()
        self.directory = directory
        self.filenames = os.listdir(os.path.join(directory, 'b'))
        self.mode = mode
        self.transformion = self.transform()

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        dess_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'b', fname))))[19:275, 19:275]
        tess_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'a', fname))))[19:275, 19:275]
        mask = np.array(Image.open(os.path.join(self.directory, 'b_mask', fname)))

        if self.mode == 'train':
            transformed = self.transformion(image=dess_img, image0=tess_img, mask=mask)
            dess_img = transformed['image']
            tess_img = transformed['image0']
            mask = transformed['mask']

        mask = torch.from_numpy(mask.astype(np.int64))

        mask = fn.one_hot(mask, 5).permute(2, 0, 1).float()# .astype(torch.float32)

        if self.mode == 'test':
            return dess_img, tess_img, mask, fname
        return dess_img, tess_img, mask

    def transform(self):
        if self.mode == 'train':
            transform = A.Compose([
                A.ElasticTransform(alpha_affine=10, sigma=0),
                A.Rotate(limit=10),
            ],
            additional_targets={'image0': 'image'})
            return transform

        elif self.mode == 'test':
            pass

    def to_zero_one(self, img):
        img = (img - img.min())/(img.max() - img.min())
        return img


    def __len__(self) -> int:
        return len(self.filenames)

class OAIDataset_womask(torch.utils.data.Dataset):
    def __init__(self, directory: str, mode: str='train'):
        '''
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        '''
        super().__init__()
        self.directory = directory
        self.filenames = os.listdir(os.path.join(directory, 'b'))
        self.mode = mode
        self.transformion = self.transform()

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        dess_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'b', fname))))[19:275, 19:275]
        tess_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'a', fname))))[19:275, 19:275]

        if self.mode == 'train':
            transformed = self.transformion(image=dess_img, image0=tess_img)
            dess_img = transformed['image']
            tess_img = transformed['image0']

        if self.mode == 'test':
            return dess_img, tess_img, fname
        return dess_img, tess_img

    def transform(self):
        if self.mode == 'train':
            transform = A.Compose([
                A.ElasticTransform(alpha_affine=10, sigma=0),
                A.Rotate(limit=10),
            ],
            additional_targets={'image0': 'image'})
            return transform

        elif self.mode == 'test':
            pass

    def to_zero_one(self, img):
        img = (img - img.min())/(img.max() - img.min())
        return img


    def __len__(self) -> int:
        return len(self.filenames)


class OAI_pain(torch.utils.data.Dataset):
    def __init__(self, directory: str, mode: str='train'):
        '''
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        '''
        super().__init__()
        self.directory = directory
        if mode == 'train':
            self.filenames = sorted(os.listdir(os.path.join(directory, 'bp')))[2000:]
        elif mode == 'test':
            # self.filenames = os.listdir(os.path.join(directory, 'bp'))[29:31] + os.listdir(os.path.join(directory, 'bp'))[34:36] +os.listdir(os.path.join(directory, 'bp'))[100:102] # 256 size from 31 # 384 size from 34
            self.filenames = sorted(os.listdir(os.path.join(directory, 'bp')))[1180:1181] + \
                             sorted(os.listdir(os.path.join(directory, 'bp')))[1182:1183] + \
                             sorted(os.listdir(os.path.join(directory, 'bp')))[1222:1223] + \
                             sorted(os.listdir(os.path.join(directory, 'bp')))[1247:1248] + \
                             sorted(os.listdir(os.path.join(directory, 'bp')))[1290:1291] + \
                             sorted(os.listdir(os.path.join(directory, 'bp')))[1367:1368]

        self.mode = mode
        self.transformion = self.transform()

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        norm_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'bp', fname))))
        pain_img = self.to_zero_one(np.array(Image.open(os.path.join(self.directory, 'ap', fname))))

        if self.mode == 'train':
            transformed = self.transformion(image=norm_img, image0=pain_img)
            norm_img = transformed['image']
            pain_img = transformed['image0']

        if self.mode == 'test':
            transformed = self.transformion(image=norm_img, image0=pain_img)
            norm_img = transformed['image']
            pain_img = transformed['image0']
            return norm_img, pain_img, fname
        return norm_img, pain_img

    def transform(self):
        if self.mode == 'train':
            transform = A.Compose([
                A.ElasticTransform(alpha_affine=10, sigma=0),
                A.Rotate(limit=10),
                A.Resize(height=256, width=256),
            ],
            additional_targets={'image0': 'image'})
            return transform

        elif self.mode == 'test':
            transform = A.Compose([
                A.Resize(height=256, width=256),
            ],
                additional_targets={'image0': 'image'})
            return transform
            # pass
    def to_zero_one(self, img):
        img = (img - img.min())/(img.max() - img.min())
        return img


    def __len__(self) -> int:
        return len(self.filenames)
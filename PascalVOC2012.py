from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from segmentation_models_pytorch.datasets import OxfordPetDataset
import os
import cv2
import albumentations as A
import numpy as np
import torch

PASCAL_VOC_PATH='/irip/chenmingxuan_2021/mmsegmentation/data/VOCdevkit/VOC2012'
DEFAULT_TRAIN_TRANSFORM = A.Compose([
    A.RandomScale(scale_limit=(0.5,1.0), p=1.0),
    A.SmallestMaxSize(max_size=512//4*5, p=1.0),
    A.RandomCrop(height=512, width=512, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0),
])

DEFAULT_TEST_TRANSFORM = A.Compose([
    A.SmallestMaxSize(max_size=512, p=1.0),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0),
])

class PascalVOC2012(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.img_path = os.path.join(PASCAL_VOC_PATH, 'JPEGImages')
        self.label_path = os.path.join(PASCAL_VOC_PATH, 'SegmentationClassAug')
        assert split in ['train', 'val', 'test']
        if split == 'train':
            data_list_file = os.path.join(PASCAL_VOC_PATH, 'ImageSets/Segmentation/trainaug.txt')
            self.transform=DEFAULT_TRAIN_TRANSFORM
        else:
            data_list_file = os.path.join(PASCAL_VOC_PATH, 'ImageSets/Segmentation/val.txt')
            self.transform=DEFAULT_TEST_TRANSFORM
        with open(data_list_file, 'r') as f:
            self.data_list = f.read().splitlines()

    def get_transform(self) -> transforms.Compose:
        return self.transform

    def set_transform(self, transform:transforms.Compose)->None:
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img = cv2.imread(os.path.join(self.img_path, img_name+'.jpg'), cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.label_path, img_name+'.png'), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img, label = transformed['image'], transformed['mask']
        return torch.from_numpy(img).permute(2,0,1), torch.from_numpy(label)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def visualize(image, mask):
        fontsize = 18
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[0].set_title('Image', fontsize=fontsize)
        ax[1].imshow(mask)
        ax[1].set_title('Mask', fontsize=fontsize)
        f.savefig('test.png')


    dataset = PascalVOC2012()
    print(len(dataset))
    print(dataset.data_list[1])
    # denorm
    # denorm = A.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225],max_pixel_value=1.0)
    img,label=dataset[1]
    img = img.permute(1,2,0).numpy()
    visualize(img, label)
    # img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('img.png', img)

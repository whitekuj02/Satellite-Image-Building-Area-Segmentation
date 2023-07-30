from torch.utils.data import Dataset
import pandas as pd
import cv2
from modules.decode import rle_decode, rle_encode
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None,infer=False, desired_mean_brightness = 127):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.desired_mean_brightness = desired_mean_brightness


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        # 이미지의 높이와 너비를 얻습니다. 2048
        height, width, _ = image.shape

        # 16분할
        split_images = [
            image[:height//4, :width//4], image[height//4:height//2, :width//4], image[height//2 :height*3//4, :width//4], image[height*3//4:, :width//4],
            image[:height//4, width//4:width//2], image[height//4:height//2, width//4:width//2], image[height//2 :height*3//4, width//4:width//2], image[height*3//4:, width//4:width//2],
            image[:height//4, width//2:width*3//4], image[height//4:height//2, width//2:width*3//4], image[height//2 :height*3//4, width//2:width*3//4], image[height*3//4:, width//2:width*3//4],
            image[:height//4, width*3//4:], image[height//4:height//2, width*3//4:], image[height//2 :height*3//4, width*3//4:], image[height*3//4:, width*3//4:]
        ]
        
        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (1024, 1024))

        # 16분할
        split_masks = [
            mask[:height//4, :width//4], mask[height//4:height//2, :width//4], mask[height//2 :height*3//4, :width//4], mask[height*3//4:, :width//4],
            mask[:height//4, width//4:width//2], mask[height//4:height//2, width//4:width//2], mask[height//2 :height*3//4, width//4:width//2], mask[height*3//4:, width//4:width//2],
            mask[:height//4, width//2:width*3//4], mask[height//4:height//2, width//2:width*3//4], mask[height//2 :height*3//4, width//2:width*3//4], mask[height*3//4:, width//2:width*3//4],
            mask[:height//4, width*3//4:], mask[height//4:height//2, width*3//4:], mask[height//2 :height*3//4, width*3//4:], mask[height*3//4:, width*3//4:]
        ]

        if self.transform:
            augmented_images = []
            augmented_masks = []
            for img, msk in zip(split_images, split_masks):
                augmented = self.transform(image=img, mask=msk)
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])
            
            split_images = augmented_images
            split_masks = augmented_masks

 
        return split_images, split_masks
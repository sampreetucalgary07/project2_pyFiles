
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import PIL
import torch


class createSampleDataset(Dataset):
    def __init__(self, img_dir, no_of_images=None, transforms=None, channel_last=False, test=False):
        self.img_dir = img_dir
        self.transform = transforms
        self.list_images = sorted(os.listdir(self.img_dir))
        if no_of_images != None:
            self.list_images = sorted(os.listdir(self.img_dir))[
                int(-1 * no_of_images):]
        if no_of_images != None and test == True:
            self.list_images = sorted(os.listdir(self.img_dir))[
                :int(no_of_images)]

            # print(self.list_images)
        self.channel_last = channel_last

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.list_images[idx])
        image = PIL.Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.channel_last:
            try:
                image = torch.permute(image, (1, 2, 0))
            except TypeError:
                print("Please convert the image to a torch tensor first! ")

        return image

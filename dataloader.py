# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomRotation, ColorJitter


class LFW4Training(Dataset):
    def __init__(self, train_file: str, img_folder: str):
        self.img_folder = img_folder

        names = os.listdir(img_folder)
        self.name2label = {name: idx for idx, name in enumerate(names)}
        self.n_label = len(self.name2label)

        with open(train_file) as f:
            train_meta_info = f.read().splitlines()

        self.train_list = []
        for line in train_meta_info:
            line = line.split("\t")
            if len(line) == 3:
                self.train_list.append(
                    (os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"), self.name2label[line[0]]))
                self.train_list.append(
                    (os.path.join(line[0], line[0] + "_" + str(line[2]).zfill(4) + ".jpg"), self.name2label[line[0]]))
            elif len(line) == 4:
                self.train_list.append(
                    (os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"), self.name2label[line[0]]))
                self.train_list.append(
                    (os.path.join(line[2], line[2] + "_" + str(line[3]).zfill(4) + ".jpg"), self.name2label[line[2]]))

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomCrop(88),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        img_path, label = self.train_list[index]  # Updated to unpack tuple directly
        img = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')  # Ensure image is RGB
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.train_list)


class LFW4Eval(Dataset):
    def __init__(self, eval_file: str, img_folder: str):
        self.img_folder = img_folder

        with open(eval_file) as f:
            eval_meta_info = f.read().splitlines()

        self.eval_list = []
        for line in eval_meta_info:
            line = line.split("\t")
            if len(line) == 3:
                eval_pair = (
                    os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"),
                    os.path.join(line[0], line[0] + "_" + str(line[2]).zfill(4) + ".jpg"),
                    1,
                )
                self.eval_list.append(eval_pair)
            elif len(line) == 4:
                eval_pair = (
                    os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"),
                    os.path.join(line[2], line[2] + "_" + str(line[3]).zfill(4) + ".jpg"),
                    0,
                )
                self.eval_list.append(eval_pair)
            else:
                pass

        self.transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        img_1_path, img_2_path, label = self.eval_list[index]
        img_1 = Image.open(os.path.join(self.img_folder, img_1_path)).convert('RGB')  # Ensure image is RGB
        img_2 = Image.open(os.path.join(self.img_folder, img_2_path)).convert('RGB')  # Ensure image is RGB
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)

        return img_1, img_2, label

    def __len__(self):
        return len(self.eval_list)
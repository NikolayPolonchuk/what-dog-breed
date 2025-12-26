import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class DogDataset(Dataset):
    def __init__(self, img_path, csv_path=None, transform=None):
        self.img_path = img_path
        self.csv_path = csv_path
        self.transform = transform
        
        self.img_names = glob.glob(f"{img_path}/*.jpg")
        
        if csv_path:
            self._load_labels()
    
    def _load_labels(self):
        label_df = pd.read_csv(self.csv_path)
        self.label_idx2name = label_df['breed'].unique()
        self.label_name2idx = {name: idx for idx, name in enumerate(self.label_idx2name)}
        
        self.img2label = {}
        for _, row in label_df.iterrows():
            img_path = f"{self.img_path}/{row['id']}.jpg"
            self.img2label[img_path] = self.label_name2idx[row['breed']]
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_path = self.img_names[index]
        
        if self.csv_path:
            label = self.img2label[img_path]
            label = torch.tensor(label)
        else:
            label = -1
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, img_names, img_dir, transform):
        self.img_names = img_names
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        name = self.img_names[idx]
        path = f"{self.img_dir}/{name}.jpg"
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, name
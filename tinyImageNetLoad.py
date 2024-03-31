import os
import shutil
from pathlib import Path
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor
import torch

path_data = Path('data')
path_data.mkdir(exist_ok=True)
path = path_data/'tiny-imagenet-200'
url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

if not path.exists():
    print("Downloading and unpacking Tiny ImageNet...")
    path_zip = shutil.urlsave(url, path_data)
    shutil.unpack_archive(str(path_data/'tiny-imagenet-200.zip'), str(path_data))

class TinyDS(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        self.files = glob(str(self.path/'**/*.JPEG'), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_path = self.files[i]
        img = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        if self.transform:
            img = self.transform(img)
        label = Path(img_path).parent.parent.name
        return img, label

class TinyValDS(TinyDS):
    def __init__(self, path, annotations_path, transform=None):
        super().__init__(path, transform=transform)
        self.anno = dict(o.split('\t')[:2] for o in annotations_path.read_text().splitlines())
        
    def __getitem__(self, i):
        img_path, label = super().__getitem__(i)
        label = self.anno[os.path.basename(img_path)]
        return img_path, label

def normalize_transform(x):
    xmean = torch.tensor([0.47565, 0.40303, 0.31555]).view(3, 1, 1)
    xstd = torch.tensor([0.28858, 0.24402, 0.26615]).view(3, 1, 1)
    return (x - xmean) / xstd

train_ds = TinyDS(path/'train', transform=normalize_transform)
val_ds = TinyValDS(path/'val', path/'val'/'val_annotations.txt', transform=normalize_transform)

dls = {
    'train': DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4),
    'val': DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)
}

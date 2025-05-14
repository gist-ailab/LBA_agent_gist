import json
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision import datasets as dset
import torchvision
from torchvision.datasets import ImageFolder
from typing import Any, Tuple
from torch.utils.data import Dataset, DataLoader
from typing import Any, Tuple, Optional
# from .aptos import APTOS2019

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(((int(image_size*1.1), int(image_size*1.1)))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config

class ImageFolderWithPaths(ImageFolder):
    """ImageFolder 확장 버전: (image, target, path) 반환"""

    def __getitem__(self, index: int) -> Tuple[Any, int, str]:
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, path

def get_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0, image_size=224):
    train_transform, test_transform = get_transform(image_size=image_size)
    train_data = ImageFolderWithPaths(path + '/train', train_transform)
    np.random.seed(seed)
    
    new_data = []
    noise_imgs = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append((train_data.samples[i][0], train_data.samples[i][1]))
        else:
            label_index = list(range(7))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append((train_data.samples[i][0], new_label))
            noise_imgs.append((train_data.samples[i][0], train_data.samples[i][1], new_label))
    train_data.samples = new_data
    train_data.targets = [label for _, label in new_data]

    # Testing
    with open('./data/label_{}.txt'.format(int(100*noise_rate)), 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))
    
    with open('./data/noise_imgs_{}.txt'.format(int(100*noise_rate)), 'w') as f:
        for i in range(len(noise_imgs)):
            f.write('{} {} {}\n'.format(noise_imgs[i][0], noise_imgs[i][1], noise_imgs[i][2]))

    valid_data = ImageFolderWithPaths(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

class ImageFolderFlexible(ImageFolder):
    """
    모드에 따라 라벨 수정/제거 기능이 있는 커스텀 ImageFolder:
    - mode='clean': 기본 GT 사용
    - mode='noise': noise_txt 기준 라벨 수정
    - mode='pred': pred_txt 기준 예측 라벨 사용
    - mode='reject': pred_txt에서 flag==1인 경우 제외
    """
    def __init__(self, root, transform=None, target_transform=None,
                 mode='clean', noise_txt: Optional[str] = None, pred_txt: Optional[str] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.mode = mode
        self.noise_map = {}
        self.pred_map = {}
        self.rejection_set = set()

        if mode == 'noise' and noise_txt:
            with open(noise_txt, 'r') as f:
                for line in f:
                    path, gt, noisy = line.strip().split(' ')
                    key = os.path.basename(path)
                    self.noise_map[key] = int(noisy)

        if mode in ('pred', 'reject') and pred_txt:
            with open(pred_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 4:
                        continue
                    path, gt, pred, flag = parts
                    fname = os.path.basename(path)
                    self.pred_map[fname] = int(pred)
                    if flag == '1':
                        self.rejection_set.add(fname)

        # samples 재구성
        new_samples = []
        for path, _ in self.samples:
            fname = os.path.basename(path)

            # reject 모드: flag=1 인 샘플 제거
            if mode == 'reject' and fname in self.rejection_set:
                continue

            # 라벨 변경
            if mode == 'noise' and fname in self.noise_map:
                label = self.noise_map[fname]
            elif mode == 'pred' and fname in self.pred_map:
                label = self.pred_map[fname]
            else:
                label = self.class_to_idx[os.path.basename(os.path.dirname(path))]

            new_samples.append((path, label))

        self.samples = new_samples
        self.targets = [label for _, label in self.samples]
        print(len(self.samples))

    def __getitem__(self, index: int) -> Tuple[Any, int, str]:
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # return image, target, path
        return image, target

def get_flexible_dataset(path, mode='clean', noise_txt=None, pred_txt=None, batch_size=32, image_size=224):
    train_transform, test_transform = get_transform(image_size=image_size)

    train_data = ImageFolderFlexible(
        root=os.path.join(path, 'train'),
        transform=train_transform,
        mode=mode,
        noise_txt=noise_txt,
        pred_txt=pred_txt
    )

    valid_data = ImageFolderFlexible(
        root=os.path.join(path, 'test'),
        transform=test_transform,
        mode='clean'  # 항상 GT 기준
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=16, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=16)

    return train_loader, valid_loader

def get_aptos_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(5))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('./data/label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_mnist_noise_dataset(dataname, noise_rate = 0.2, batch_size = 32, seed = 0):
    # from medmnist import NoduleMNIST3D
    from medmnist import PathMNIST, BloodMNIST, OCTMNIST, TissueMNIST, OrganCMNIST
    train_transform, test_transform = get_transform()

    if dataname == 'pathmnist':
        train_data = PathMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = PathMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 9
    if dataname == 'bloodmnist':
        train_data = BloodMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = BloodMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'octmnist':
        train_data = OCTMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OCTMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 4
    if dataname == 'tissuemnist':
        train_data = TissueMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = TissueMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'organcmnist':
        train_data = OrganCMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OrganCMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 11

    np.random.seed(seed)
    # new_imgs = []
    new_labels =[]
    for i in range(len(train_data.imgs)):
        if np.random.rand() > noise_rate: # clean sample:
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(train_data.labels[i][0])
        else:
            label_index = list(range(num_classes))
            label_index.remove(train_data.labels[i])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(new_label)
    # train_data.imgs = new_imgs
    train_data.labels = new_labels

    new_labels = []
    for i in range(len(test_data.labels)):
        new_labels.append(test_data.labels[i][0])
    test_data.labels = new_labels

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

# def read_noise_datalist(path, batch_size=1, seed=0):
#     with open(path, 'r') as f:

class NoiseDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (str): noise_imgs.txt 파일의 경로.
                각 줄이 "<img_path> <origin_label> <noise_label>" 형태로 기록되어 있음.
            transform (callable, optional): 이미지에 적용할 변환 함수.
        """
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue  # 형식이 올바르지 않은 줄은 건너뜁니다.
                img_path, origin_label, noise_label = parts
                self.samples.append((img_path, int(origin_label), int(noise_label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, origin_label, noise_label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 평가 시에 모델의 예측과 함께 원래 라벨과 노이즈 라벨 모두를 활용할 수 있도록 반환합니다.
        return image, origin_label, noise_label

# noise_imgs.txt 파일에 기록된 데이터만 로드하는 dataloader를 반환하는 함수
def read_noise_datalist(path, batch_size=32, num_workers=8):
    """
    Args:
        path (str): noise_imgs.txt 파일이 위치한 디렉토리의 경로.
        batch_size (int): 배치 사이즈.
        num_workers (int): DataLoader에서 사용하는 worker 수.
    Returns:
        DataLoader: noise_imgs.txt에서 읽은 데이터를 로드하는 DataLoader.
    """
    # 테스트 시에 사용할 transform을 가져옵니다.
    _, test_transform = get_transform()
    noise_txt = path + '/noise_imgs.txt'
    dataset = NoiseDataset(noise_txt, transform=test_transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,     # 평가에서는 보통 순서를 그대로 유지합니다.
                        pin_memory=True,
                        num_workers=num_workers)
    return loader

def get_clean_dataset(path, batch_size = 32):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_clean_aptos_dataset(path, batch_size = 32):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)
    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

if __name__=='__main__':
    # 1. Clean 모드 (기본 GT 라벨)
    train_loader, val_loader = get_flexible_dataset(
        path='./dataset/ham10000', mode='clean', batch_size=32
    )
    
    # 2. Noise 모드 (노이즈 라벨 적용)
    train_loader, val_loader = get_flexible_dataset(
        path='./dataset/ham10000', mode='noise', 
        noise_txt='./data/noise_imgs.txt', batch_size=32
    )

    # 3. Pred 모드 (모델 예측 라벨 사용)
    train_loader, val_loader = get_flexible_dataset(
        path='./dataset/ham10000', mode='pred', 
        pred_txt='./data/ham10000_train_data_noise_label_list.txt', batch_size=32
    )

    # 4. Reject 모드 (flag==1 샘플 제거)
    train_loader, val_loader = get_flexible_dataset(
        path='./dataset/ham10000', mode='reject', 
        pred_txt='./data/ham10000_train_data_noise_label_list.txt', batch_size=32
    )
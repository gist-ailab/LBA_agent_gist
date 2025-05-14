import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import utils
import rein
import dino_variant
from utils import ece, ace, tace
from tqdm import tqdm
import random
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import glob
import data.loader
import data.loader_ham10000
import cv2
import json
import random
import timm


def get_outputs(model, loader, device, mode='rein'):
    model.eval()
    all_probs = []
    all_labels = []

    def forward_fn(model, inputs):
        if mode == 'rein':
            f = model.forward_features(inputs)
            f = f[:, 0, :]
            return model.linear_rein(f)
        elif mode == 'no_rein':
            f = model.forward_features_no_rein(inputs)
            f = f[:, 0, :]
            return model.linear(f)
        else:
            f = model(inputs)
            return model.linear(f)

    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader):
            inputs = inputs.to(device)
            outputs = forward_fn(model, inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)

def evaluate(model, loader, device, mode='rein'):
    acc = utils.validation_accuracy(model, loader, device, mode)
    probs, labels = get_outputs(model, loader, device, mode)
    ece_score = ece(labels, probs)
    ace_score = ace(labels, probs)
    tace_score = tace(labels, probs, threshold=0.01)
    return acc, ece_score, ace_score, tace_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--train_mode', type=str, choices=['up_ext', 'up_com', 'down_ext', 'down_com', 'ham10000'])
    parser.add_argument('--model', type=str, required=True, choices=['DINO_s', 'DINO_b', 'DINO_l'])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
    args = parser.parse_args()

    device = 'cuda:'+args.gpu_id
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transform = albumentations.Compose([
                    albumentations.Resize(args.img_size,args.img_size),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.Rotate(limit=(-30,30), p=0.5),
                    albumentations.ShiftScaleRotate(shift_limit=(-0.15, 0.15), scale_limit=1.0, rotate_limit=0, p=0.5)
                    ])

    transform2 = albumentations.Compose([
                    albumentations.Resize(args.img_size,args.img_size),
                    ])
    if args.train_mode=='ham10000':
        num_classes = 7
    else:
        num_classes = 4 if args.train_mode=='down_ext' else 3
    img_folder = 'max' if 'up' in args.train_mode else 'man'

    train_imgs = glob.glob("./dataset/crop_PNG_Images/train_{}/*.png".format(img_folder))
    val_imgs = glob.glob("./dataset/crop_PNG_Images/val_{}/*.png".format(img_folder))
    test_imgs = glob.glob("./dataset/crop_PNG_Images/test_{}/*.png".format(img_folder))

    train_dir = train_imgs + val_imgs
    test_dir = test_imgs
    
    if 'up' in args.train_mode:
        if 'ext' in args.train_mode:
            train_dataset = loader_up.UpDataLoader_ext(train_dir, set="train", he=args.he_type, transform=transform)
            train_dataset = loader_up.NoisyUpDataLoader(train_dataset, num_classes, args.noise_rate)
            test_dataset = loader_up.UpDataLoader_ext(test_dir, set="test", he=args.he_type, transform=transform2)
        else:
            train_dataset = loader_up.UpDataLoader_com(train_dir, set="train", he=args.he_type, transform=transform)
            train_dataset = loader_up.NoisyUpDataLoader(train_dataset, num_classes, args.noise_rate)
            test_dataset = loader_up.UpDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size = args.batch_size ,shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)
    elif 'down' in args.train_mode:
        if 'ext' in args.train_mode:
            train_dataset = loader_down.DownDataLoader_ext(train_dir, set="train", he=args.he_type, transform=transform)
            train_dataset = loader_down.NoisyDownDataLoader(train_dataset, num_classes, args.noise_rate)
            test_dataset = loader_down.DownDataLoader_ext(test_dir, set="test", he=args.he_type, transform=transform2)
        else:
            train_dataset = loader_down.DownDataLoader_com(train_dir, set="train", he=args.he_type, transform=transform)
            train_dataset = loader_down.NoisyDownDataLoader(train_dataset, num_classes, args.noise_rate)
            test_dataset = loader_down.DownDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size = args.batch_size ,shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)

    elif 'ham10000' in args.train_mode:
        train_loader, test_loader = loader_ham10000.get_noise_dataset(path='./dataset/ham10000', 
                                                                    noise_rate=args.noise_rate, 
                                                                    batch_size=args.batch_size,
                                                                    seed=args.seed,
                                                                    image_size=args.img_size)

    # 모델 정의 및 로드
    if args.model == 'DINO_l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear_rein = torch.nn.Linear(variant['embed_dim'], num_classes)
        model.in_chans = 3
        model.load_state_dict(torch.load(args.weight_path), strict=False)
    elif args.model == 'DINO_b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear_rein = torch.nn.Linear(variant['embed_dim'], num_classes)
        model.in_chans = 3
        model.load_state_dict(torch.load(args.weight_path), strict=False)
    elif args.model == 'DINO_s':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear_rein = torch.nn.Linear(variant['embed_dim'], num_classes)
        model.in_chans = 3
        
        checkpoint = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.to(device)
    model.eval()

    # 평가 수행
    acc, ece_score, ace_score, tace_score = evaluate(model, test_loader, device, mode='rein')
    print(f"[Eval] Acc: {acc:.4f}, ECE: {ece_score:.4f}, ACE: {ace_score:.4f}, TACE: {tace_score:.4f}")

if __name__ == '__main__':
    main()
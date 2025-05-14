from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
from PIL import Image
import data.loader
import data.loader_ham10000
import glob
import cv2
import torch.optim as optim
from tqdm import tqdm
import time
import torchvision.models as models
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import json
import random
import timm
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler
# from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import warnings
import torch.nn.functional as F
import rein
import dino_variant
import utils
from rein.models.backbones.dino_layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    NestedTensorBlock as Block,
)

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
# Learning
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--gpu_id', type=str ,default='4')
parser.add_argument('--train_mode', type=str, choices=['up_ext', 'up_com', 'down_ext', 'down_com', 'ham10000'])
parser.add_argument("--model", choices=["Vision_Transformer", "resnet34", "resnet152d","seresnet101", "efficientnet_b3", "R50-ViT-B_16","coat","efficientnet_l2", "DINO_s", "DINO_b", "DINO_l"],
                    default="DINO_l",help="Which model to use.")
# parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16","ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "DINO_s", "DINO_b", "DINO_l"],
#                     default="DINO_l",help="Which variant to use.")
parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
parser.add_argument("--pretrained_dir", type=str, default="Vit_pretrain/imagenet21k/imagenet21k_ViT-B_16.npz", 
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--he_type', type=str, default='CLAHE', choices=['CLAHE', 'HE'])
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD','AdamW'])
#CBAM
parser.add_argument('--depth', type=int, default=50, choices=[18,34,50,101])
parser.add_argument('--att_type', type=str, default='CBAM', choices=['CBAM','BAM','basic'])
args = parser.parse_args()


save_path = "./checkpoints/{}/checkpoint_model_{}_lr_{}_batchsize_{}_clean_{}"\
    .format(args.train_mode, args.model, args.learning_rate, args.batch_size, int(time.time()))
    
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
device = 'cuda:'+args.gpu_id
lr_decay = [int(0.5*args.epoch_num), int(0.75*args.epoch_num), int(0.9*args.epoch_num)]

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

if 'ham10000' in args.train_mode:
    num_classes = 7
    train_loader, test_loader = loader_ham10000.get_flexible_dataset(
        path='./dataset/ham10000', 
        mode='reject', 
        noise_txt='/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/noise_imgs.txt',
        pred_txt='/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/ham10000_train_data_noise_label_list.txt', 
        batch_size=32
    )
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
            test_dataset = loader_up.UpDataLoader_ext(test_dir, set="test", he=args.he_type, transform=transform2)
        else:
            train_dataset = loader_up.UpDataLoader_com(train_dir, set="train", he=args.he_type, transform=transform)
            test_dataset = loader_up.UpDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size = args.batch_size ,shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)
    elif 'down' in args.train_mode:
        if 'ext' in args.train_mode:
            train_dataset = loader_down.DownDataLoader_ext(train_dir, set="train", he=args.he_type, transform=transform)
            test_dataset = loader_down.DownDataLoader_ext(test_dir, set="test", he=args.he_type, transform=transform2)
        else:
            train_dataset = loader_down.DownDataLoader_com(train_dir, set="train", he=args.he_type, transform=transform)
            test_dataset = loader_down.DownDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size = args.batch_size ,shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)

if args.model == 'DINO_s':
    model_load = dino_variant._small_dino
    variant = dino_variant._small_variant
    print(model_load)
    basemodel = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = basemodel.state_dict()
    
    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], num_classes)
    model.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model.in_chans=3
    
elif args.model == 'DINO_b':
    model_load = dino_variant._base_dino
    variant = dino_variant._base_variant
    print(model_load)
    basemodel = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = basemodel.state_dict()
    
    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], num_classes)
    model.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model.in_chans=3
    
elif args.model == 'DINO_l':
    model_load = dino_variant._large_dino
    variant = dino_variant._large_variant
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/LBA_third_molar_noise_detection/checkpoints/pretrained_384/dinov2_vitl16_pretrain.pth')
    
    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], num_classes)
    model.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model.in_chans=3
    model.eval()

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 

print('## Trainable parameters')
model.train()
for n, p in model.named_parameters():
    if p.requires_grad == True:
        print(n)
        
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, saver, device):
    avg_accuracy = 0.0
    for epoch in range(args.epoch_num):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        # correct_linear = 0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            loss_rein = criterion(outputs, targets)
            loss = loss_rein.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       
 
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc1: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, test_loader, device)
        
        scheduler.step()
        if epoch >= args.epoch_num-10:
            avg_accuracy += valid_accuracy 
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}],  VALID_1 [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))

def main():
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, saver, device)
    print(save_path)

if __name__=='__main__':
    main()
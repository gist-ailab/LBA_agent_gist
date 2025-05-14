from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
import data.loader_thirdmolar
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
parser.add_argument('--noise_rate', type=float, default=0.1)
parser.add_argument('--low', type=float, default=0.4)
parser.add_argument('--high', type=float, default=0.5)
parser.add_argument('--duration', type=float, default=10)
args = parser.parse_args()


save_path = "./checkpoints/{}/checkpoint_model_{}_lr_{}_batchsize_{}_noise_rate_{}_{}"\
    .format(args.train_mode, args.model, args.learning_rate, args.batch_size, args.noise_rate, int(time.time()))
    
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
    
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model2.in_chans=3
    
    model.to(device)
    model2.to(device)
    model.eval()
    model2.eval()
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
    
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model2.in_chans=3
    
    model.to(device)
    model2.to(device)
    model.eval()
    model2.eval()
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
    
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear_rein = nn.Linear(variant['embed_dim'], num_classes)
    model2.in_chans=3
    model2.eval()

model = model.to(device)
model2 = model2.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)
saver = timm.utils.CheckpointSaver(model2, optimizer, checkpoint_dir= save_path, max_history = 1) 

print('## Trainable parameters')
model2.train()
for n, p in model2.named_parameters():
    if p.requires_grad == True:
        print(n)
        
def train_with_noise_label(model, model2, train_loader, test_loader, criterion, optimizer, optimizer2, scheduler, scheduler2, saver, device):
    avg_accuracy = 0.0
    for epoch in range(args.epoch_num):
        ## training
        model.train()
        model2.train()
        total_loss = 0
        total = 0
        correct = 0
        correct2 = 0
        correct_linear = 0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            features_rein2 = model2.forward_features(inputs)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model2.linear_rein(features_rein2)

            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)

                pred2 = outputs.max(1).indices
                linear_accurate2 = (pred2==targets)

            loss_linear = criterion(outputs_, targets)
            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_rein2 = linear_accurate2*criterion(outputs2, targets)
            
            loss = loss_linear.mean()+loss_rein.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            optimizer2.zero_grad()
            loss_rein2.mean().backward()
            optimizer2.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs2[:len(targets)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct2/total, 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model2, test_loader, device)
        valid_accuracy_ = utils.validation_accuracy(model, test_loader, device)
        valid_accuracy_linear = utils.validation_accuracy(model, test_loader, device, mode='no_rein')
        
        scheduler.step()
        scheduler2.step()
        if epoch >= args.epoch_num-10:
            avg_accuracy += valid_accuracy 
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))

def train_with_noise_label_refine(model, model2, train_loader, test_loader, criterion, optimizer, optimizer2, scheduler, scheduler2, saver, device):
    avg_accuracy = 0.0
    for epoch in range(args.epoch_num):
        ## training
        model.train()
        model2.train()
        total_loss = 0
        total = 0
        correct = 0
        correct2 = 0
        correct_linear = 0
        sum_linear1 = 0
        sum_linear2 = 0
        sum_changed1 = 0
        sum_changed2 = 0
        step = (args.high - args.low) / args.duration
        decay_step = epoch // (args.epoch_num// args.duration)
        mix_ratio = args.high - step * decay_step
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            features_rein2 = model2.forward_features(inputs)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model2.linear_rein(features_rein2)

            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                targets_ = targets.clone()
                softmax_outputs_ = F.softmax(outputs_, dim=1)
                pred = softmax_outputs_.max(1)
                pred_indices = pred.indices
                pred_confidences = pred.values
                # if epoch >= args.refine_epoch:
                old_targets_ = targets_
                targets_ = mix_ratio*F.one_hot(targets_, num_classes=num_classes).to(targets.device) + (1-mix_ratio)*softmax_outputs_
                targets_dist = torch.softmax(targets_, dim=-1)
                targets_ = targets_dist.max(1).indices
                changed = (old_targets_ != targets_)
                sum_changed1 += sum(changed)
                linear_accurate = (pred_indices==targets_)
                sum_linear1 += sum(linear_accurate)

                targets__ = targets.clone()
                softmax_outputs = F.softmax(outputs, dim=1)
                pred2 = softmax_outputs.max(1)
                pred2_indices = pred2.indices
                pred2_confidences = pred2.values
                # if epoch >= args.refine_epoch:
                old_targets__ = targets__
                targets__ = mix_ratio*F.one_hot(targets__, num_classes=num_classes).to(targets.device) + (1-mix_ratio)*softmax_outputs
                targets__dist = torch.softmax(targets__, dim=-1)
                targets__ = targets__dist.max(1).indices
                changed = (old_targets__ != targets__)
                sum_changed2 += sum(changed)
                linear_accurate2 = (pred2_indices==targets__)
                sum_linear2 += sum(linear_accurate2)
            
            loss_linear = criterion(outputs_, targets)
            loss_rein = linear_accurate*utils.cross_entropy_soft_label(outputs, targets_dist)
            loss_rein2 = linear_accurate2*utils.cross_entropy_soft_label(outputs2, targets__dist)
            
            loss = loss_linear.mean()+loss_rein.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            optimizer2.zero_grad()
            loss_rein2.mean().backward()
            optimizer2.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets_)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs2[:len(targets__)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d) | LA1 : %d | LA2 : %d | Ch1 : %d | Ch2 : %d'
                        % (total_loss/(batch_idx+1), 100.*correct2/total, 100.*correct/total, 100.*correct_linear/total, correct, total, sum_linear1, sum_linear2, sum_changed1, sum_changed2), end = '')                      
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model2, test_loader, device)
        valid_accuracy_ = utils.validation_accuracy(model, test_loader, device)
        valid_accuracy_linear = utils.validation_accuracy(model, test_loader, device, mode='no_rein')
        
        scheduler.step()
        scheduler2.step()
        if epoch >= args.epoch_num-10:
            avg_accuracy += valid_accuracy 
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))

def generate_label_noise_list(model2, data_loader, device):
    save_path = utils.gen_label_noise_list(model2, data_loader, device, args.train_mode, mode='rein', noise_rate=args.noise_rate)
    return save_path

def main():
    # train_with_noise_label(model, model2, train_loader, test_loader, criterion, optimizer, optimizer2, scheduler, scheduler2, saver, device)
    train_with_noise_label_refine(model, model2, train_loader, test_loader, criterion, optimizer, optimizer2, scheduler, scheduler2, saver, device)
    save_path = generate_label_noise_list(model2, train_loader, device)
    print(save_path)

if __name__=='__main__':
    main()
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import contextlib
import io
import sys
sys.path.append("/SSDe/youmin_park/adapter-weight-ensemble/")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import argparse
import numpy as np
import glob
from torchvision import models
from torch.cuda.amp.autocast_mode import autocast
from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, validation_accuracy_lora, compute_aurc, compute_auroc, compute_fpr95
from data import dataloader
import rein


def initialize_model(config, device, args):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])  
    model = model.to(device)
    return model


def get_model_from_sd(state_dict, config, device, args):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    return model

# Greedy soup model ensembling
def greedy_soup_ece(models, model_names, valid_loader, device, config, args):
    # Calculate ECE for each model and sort them by ECE in ascending order (lower ECE is better)
    ece_list = [validate(model, valid_loader, device, args) for model in models]
    # print("ECE for each model:")
    # print(ece_list)
    model_ece_pairs = [(model, ece, name) for model, ece, name in zip(models, ece_list, model_names)]
    sorted_models = sorted(model_ece_pairs, key=lambda x: x[1])
    
    print("Sorted models with ECE performance:")
    for model, ece, name in sorted_models:
        print(f'Model: {name}, ECE: {ece}')

    best_ece = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()
    greedy_soup_ingredients = [sorted_models[0][0]]
    
    TOLERANCE = (sorted_models[-1][1] - sorted_models[0][1]) / 2
    TOLERANCE = 0

    print(f'Tolerance: {TOLERANCE}')

    for i in range(1, len(models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')
        
        # Calculate potential new parameters with the new ingredient
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) + 
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }

        temp_model = get_model_from_sd(potential_greedy_soup_params, config, device, args)
        temp_model.eval()
        
        # Evaluate the potential greedy soup model
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                output = torch.softmax(output, dim=1)
        
                outputs.append(output.cpu())
                targets.append(target.cpu())    
        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy().astype(int)
        held_out_val_ece = calculate_ece(outputs, targets)
        
        print(f'Potential greedy soup ECE: {held_out_val_ece}, best ECE so far: {best_ece}.')
        
        # Add new ingredient to the greedy soup if it improves ECE or is within tolerance
        if held_out_val_ece < best_ece + TOLERANCE:
            best_ece = held_out_val_ece
            greedy_soup_ingredients.append(sorted_models[i][0])
            greedy_soup_params = potential_greedy_soup_params
            print(f'<Added new ingredient to soup. Total ingredients: {len(greedy_soup_ingredients)}>\n')
        else:
            print(f'<No improvement. Reverting to best-known parameters.>\n')


    final_model = get_model_from_sd(greedy_soup_params, config, device, args)
        
    return greedy_soup_params, final_model


def greedy_soup_acc(models, model_names, valid_loader, device, config, args):
    # Evaluate and sort models by validation accuracy
    model_accuracies = [(model, validation_accuracy(model, valid_loader, device, mode = 'resnet'), name) for model, name in zip(models, model_names)]
    
    # Sort models based on accuracy
    sorted_models = sorted(model_accuracies, key=lambda x: x[1], reverse=True)
    
    # Print sorted models with their names and accuracies
    print("Sorted models by accuracy:")
    for model, acc, name in sorted_models:
        print(f'Model: {name}, Accuracy: {acc}')
    print("\n")
    
    # Initialize greedy soup with the highest-performing model
    max_accuracy = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()  # Best model's initial parameters
    greedy_soup_ingredients = [sorted_models[0][0]] 

    for i in range(1, len(sorted_models)):
        print(f'Testing model {i+1} ({sorted_models[i][2]}) of {len(sorted_models)}')
        
        # previous_greedy_soup_params = {k: v.clone() for k, v in greedy_soup_params.items()}
        
        # New model parameters to test as an additional ingredient
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')    
    
        # Create potential new soup parameters by averaging with the new ingredient
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1)) +
               new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }
        
        # Load the new potential parameters into the base model for evaluation
        temp_model = get_model_from_sd(potential_greedy_soup_params, config, device, args)
        temp_model.eval()
        
        # Calculate validation accuracy with the potential new soup parameters
        held_out_val_accuracy = validation_accuracy(temp_model, valid_loader, device, mode = 'resnet')
        
        print(f'Held-out validation accuracy: {held_out_val_accuracy}, best accuracy so far: {max_accuracy}.\n')
        
        # Update greedy soup if accuracy improves, otherwise revert to original parameters
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i][0])
            max_accuracy = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params  # Save the improved parameters
            print(f'[New greedy soup ingredient added. Number of ingredients: {len(greedy_soup_ingredients)}]\n')
        else:
            print(f'[No improvement. Reverting to best-known parameters.]\n')
         
        final_model = get_model_from_sd(greedy_soup_params, config, device, args)
        

    return greedy_soup_params, final_model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar100')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    parser.add_argument('--checkpoint', '-c', type=str)
    parser.add_argument('--soup', '-s', type=str, default='acc')
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    # batch_size = int(config['batch_size'])
    batch_size = 128
    checkpoint = args.checkpoint
    num_workers = int(config['num_workers'])
    
    save_paths = [ 
        # os.path.join(config['save_path'], checkpoint, 'cyclic_checkpoint_epoch219.pth'),
        # os.path.join(config['save_path'], checkpoint, 'cyclic_checkpoint_epoch249.pth'),
    ]
    
    
    checkpoint_dir = os.path.join(config['save_path'], checkpoint)
    save_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "cyclic_checkpoint_epoch*.pth")))

    # print(save_paths) 
    print(f'Found {len(save_paths)} models to soup.')
    
    
    model_names = [os.path.basename(path) for path in save_paths]

    
    models = []

    
    for save_path in save_paths:
        model = initialize_model(config, device, args)
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True) # 수정
        model.to(device)
        model.eval()
        models.append(model)

    
    _, valid_loader, test_loader = dataloader.setup_data_loaders(args, data_path, batch_size)
    
    if args.soup == 'acc':
        print('Greedy soup by ACC')
        greedy_soup_params, model = greedy_soup_acc(models, model_names, valid_loader, device, config, args)
    elif args.soup == 'ece':
        print('Greedy soup by ECE')
        greedy_soup_params, model = greedy_soup_ece(models, model_names, valid_loader, device, config, args)
    

    model = get_model_from_sd(greedy_soup_params, config, device, args)
    model.eval()
    

    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print("\n🔹 Accuracy Metrics 🔹")
    print('test acc:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            output = torch.softmax(output, dim=1)
                    # print(output.shape)
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()

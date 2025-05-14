# inference_evaluation.py

import torch
import numpy as np
from .evaluation import calculate_ece
from torch.cuda.amp.autocast_mode import autocast

def rein_forward(model, inputs):
    # Forward pass for the model
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        output = torch.softmax(output, dim=1)
    return output

def resnet_forward(model, inputs):
    output = model(inputs)
    output = torch.softmax(output, dim=1)
    return output

def adaptformer_forward(model, inputs):
    f = model.forward_features(inputs)
    outputs = model.linear(f)
    outputs = torch.softmax(outputs, dim=1) 
    return outputs


def validate(model, valid_loader, device, args):
    """
    Perform inference on the test_loader using the given model and evaluate results.

    Args:
    - model: The model to use for inference.
    - test_loader: DataLoader for the test set.
    - device: Device to perform inference on (e.g., 'cpu' or 'cuda').
    - evaluation: Evaluation module with evaluate function.

    Returns:
    - None, but prints evaluation results.
    """
    outputs = []
    targets = []
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, target in valid_loader:
            inputs, target = inputs.to(device), target.to(device)
            # if args.adapter == 'linear':
            #     output = model(inputs)
                # output = torch.softmax(output, dim=1)
            if args.adapter == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)  
            elif args.adapter == 'lora':
                with autocast(enabled=True):
                    output = lora_forward(model, inputs)
            elif args.adapter == 'adaptformer':
                output = adaptformer_forward(model, inputs)
            # elif args.adapter == 'resnet':
            #     output = resnet_forward(model, inputs)

            
            # Append results
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    # Concatenate all outputs and targets
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    
    # Call the evaluation function
    ece = calculate_ece(outputs, targets)
    
    
    return ece

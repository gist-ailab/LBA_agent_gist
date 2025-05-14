import torch
import torch.nn.functional as F
# from calibration import ece, rmsce, sce, ace, tace

recall_level_default = 0.95

def cross_entropy_soft_label(pred_logits, soft_targets, reduction='none'):
    """
    Cross Entropy loss that supports soft targets.

    Args:
        pred_logits (Tensor): (B, C) logits output from the model (before softmax).
        soft_targets (Tensor): (B, C) soft labels (e.g. with label smoothing).
        reduction (str): 'none' | 'mean' | 'sum'
    
    Returns:
        loss (Tensor)
    """
    log_probs = F.log_softmax(pred_logits, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1)  # shape: (B,)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction

def validation_accuracy_lora(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1) 
            outputs = model.forward_features(inputs)
            outputs = model.linear(outputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    # print("\n🔹 Accuracy Metrics 🔹")
    return valid_accuracy


def validation_accuracy(model, loader, device, mode='rein'):
    total = 0
    correct = 0

    # Output functions for different modes
    def linear(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs
    
    # def rein(model, inputs):
    #     # ModelWithTemperature로 래핑된 경우 내부 모델에 접근
    #     if isinstance(model, ModelWithTemperature):
    #         f = model.model.forward_features(inputs)  # 내부 모델의 forward_features 호출
    #         f = f[:, 0, :]
    #         outputs = model.model.linear(f)  # 내부 모델의 linear 레이어 호출
    #     else:
    #         f = model.forward_features(inputs)  # 이미 원래 모델인 경우 그대로 호출
    #         f = f[:, 0, :]  # CLS 토큰만 선택
    #         outputs = model.linear(f)  # Linear 레이어 적용
        
    #     return outputs
    
    def rein(model, inputs):
        f = model.forward_features(inputs)  # 이미 원래 모델인 경우 그대로 호출
        f = f[:, 0, :]  # CLS 토큰만 선택
        outputs = model.linear(f)  # Linear 레이어 적용
        
        return outputs
    
    def adaptformer(model, inputs):
        f = model.forward_features(inputs)
        outputs = model.linear(f)
        return outputs

    
    def rein3(model, inputs):
        f = model.forward_features1(inputs)
        f = f[:, 0, :]
        outputs1 = model.linear(f)

        f = model.forward_features2(inputs)
        f = f[:, 0, :]
        outputs2 = model.linear(f)

        f = model.forward_features3(inputs)
        f = f[:, 0, :]
        outputs3 = model.linear(f)
        return outputs1 + outputs2 + outputs3

    def rein_dropout(model, inputs, num_samples=10):
        model.train() 
        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                f = model.forward_features(inputs)
                f = f[:, 0, :]
                output = model.linear(f)
                outputs.append(torch.softmax(output, dim=1))

        output = torch.mean(torch.stack(outputs), dim=0)
        return output

    def no_rein(model, inputs):
        f = model.forward_features_no_rein(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs
    
    def resnet(model, inputs):
        outputs = model(inputs)
        return outputs
    
    def densenet(model, inputs):
        outputs = model(inputs)
        return outputs
    
    def vgg(model, inputs):
        outputs = model(inputs)
        return outputs

    # Select appropriate output function
    if mode == 'rein':
        out = rein
    elif mode == 'no_rein':
        out = no_rein
    elif mode == 'adaptformer':
        out = adaptformer
    elif mode == 'rein3':
        out = rein3
    elif mode == 'rein_dropout':
        out = rein_dropout
        model.train()
    elif mode == 'resnet':
        out = resnet
    elif mode == 'densenet':
        out = densenet
    elif mode == 'vgg':
        out = vgg
    else:
        out = linear

    if mode == 'rein_dropout':
        model.train()
    else:
        model.eval()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1)
            
            outputs = out(model, inputs)
            _, predicted = outputs.max(1)

            # Ensure the same batch size for predicted and targets
            min_batch_size = min(predicted.size(0), targets.size(0))
            predicted = predicted[:min_batch_size]
            targets = targets[:min_batch_size]

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    valid_accuracy = correct / total
    return valid_accuracy
    

def gen_label_noise_list(model, loader, device, train_mode, mode='rein', noise_rate=0.2):
    total = 0
    correct = 0
    
    def linear(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs
    
    def rein(model, inputs):
        f = model.forward_features(inputs)
        f = f[:, 0, :]
        outputs = model.linear_rein(f)
        return outputs
    
    def no_rein(model, inputs):
        f = model.forward_features_no_rein(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs
    
    if mode == 'rein':
        out = rein
    elif mode == 'no_rein':
        out = no_rein
    else:
        out = linear

    save_path = f"./data/{train_mode}_train_data_noise_label_{noise_rate}_list.txt"
    
    model.eval()
    with open(save_path, 'w') as f:
        with torch.no_grad():
            for batch_idx, (inputs, targets, img_name) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = out(model, inputs)
                _, predicted = outputs.max(1)
                
                # is_noise = 1 if prediction and target do not match
                is_noise = (predicted != targets).long()  # (B,) tensor

                for i in range(inputs.size(0)):
                    line = f"{img_name[i]} {targets[i].item()} {predicted[i].item()} {is_noise[i].item()}\n"
                    f.write(line)
    return save_path
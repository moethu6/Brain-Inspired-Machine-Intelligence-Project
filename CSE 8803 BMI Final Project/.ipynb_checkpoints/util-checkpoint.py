# +
import torch
import numpy as np
import pickle

from torch import nn
from collections import OrderedDict

# +
def load_model(model, save_path, strict=True):
    '''Load a previously saved model for inference'''
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model

def load_model_and_opt(model, opt, save_path, strict=True):
    '''Load a previously saved model and optimizer for more training'''
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, opt


# -

def save_model(save_path, model, optimizer, epoch):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)


def batch_predict(model, data, T=None, batch_size=512, fw_pass_fn=None, device='cpu'):
    '''
    Predict classes for the entire dataset.
    If the data doesn't fit on GPU memory, we need to split it into batches to make predictions.
    
    fw_pass_fn: (optional) function that computes the forward pass for SNNs, i.e. if the model needs
    to be called for each time step, this function should do it and return the averaged results 
    '''
    pred = []
    model.eval()
    
    with torch.no_grad():
    #     fw_pass_fn = (model if fw_pass_fn is None else fw_pass_fn)
        for b in range(np.ceil(len(data) / batch_size).astype(int)):
            batch = data[b*batch_size : (b+1)*batch_size]
            if T:
                batch_pred = fw_pass_fn(model, T, torch.stack(batch).squeeze().to(device)).detach().cpu()
            else:
                batch_pred = model(torch.stack(batch).squeeze().to(device)).detach().cpu()
            pred.append(batch_pred)
        return torch.cat(pred, dim=0) if T is None else torch.cat(pred, dim=1)


def load_from_cnn(model, cnn_model_weights, activations_path, num_lif=4):
    checkpoint = torch.load(cnn_model_weights, weights_only=True)
    weights = checkpoint['model_state_dict']
    
    act_file = open(activations_path, 'rb')
    max_act = pickle.load(act_file)
    act_file.close()

    conv_weights = OrderedDict()
    conv_weights['inhibitor'] = nn.init.kaiming_uniform_(torch.empty(91, 91))
    conv_weights['embedding.weight'] = torch.from_numpy(wv.vectors)
    skip_names = ('embedding', 'relu', 'flatten')
    
    layer_names = list(k for k in max_act.keys() if not any(skip_name in k for skip_name in skip_names))
    for i, layer_name in enumerate(layer_names):
        if 'lstm' in layer_name:
            conv_weights[f'{layer_name}.weight_ih_l0'] = weights[f'{layer_name}.weight_ih_l0'] * (1. / max_act[layer_name])
            conv_weights[f'{layer_name}.weight_hh_l0'] = weights[f'{layer_name}.weight_hh_l0'] * (1. / max_act[layer_name])
            conv_weights[f'{layer_name}.bias_ih_l0'] = weights[f'{layer_name}.bias_ih_l0'] / max_act[layer_name]
            conv_weights[f'{layer_name}.bias_hh_l0'] = weights[f'{layer_name}.bias_hh_l0'] / max_act[layer_name]
        else:
            conv_weights[f'{layer_name}.weight'] = weights[f'{layer_name}.weight'] * (max_act[layer_names[i-1]] / max_act[layer_name])
    
    # SNN-specific weights
    for i in range(num_lif):
        conv_weights[f'lif{i}.beta'] = torch.tensor(beta)
        conv_weights[f'lif{i}.threshold'] = torch.tensor(1.)
        conv_weights[f'lif{i}.graded_spikes_factor'] = torch.tensor(1.)
        conv_weights[f'lif{i}.reset_mechanism_val'] = torch.tensor(0)
    
    model.load_state_dict(conv_weights)
    print('Loaded weights from pre-trained CNN successfully')
    return model

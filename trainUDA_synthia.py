import argparse
import os
import sys
import random
import timeit
import datetime
import cv2

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from configs.global_vars import IMG_MEAN
from model.deeplabv2_synthia import Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
from utils.loss import BinaryDiceLoss

from data import get_loader, get_data_path, cityscapesLoader
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate

import time
from matplotlib import pyplot as plt
from utils.visualization import subplotimg

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()

def seed_worker(worker_id):
    # Taken from https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda() 
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  

    return criterion(pred, label)


def loss_edge_calc(pred, label):
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(BinaryDiceLoss(), device_ids=gpus).cuda() 
    else:
        criterion = BinaryDiceLoss().cuda()
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def create_ema_model(model):
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    
    if len(gpus)>1:
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def main():
    print(config)

    best_mIoU = 0

    if consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss =  torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()
        else:
            unlabeled_loss =  MSELoss2d().cuda()
    elif consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label), device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if config['pretrained'] == 'imagenet':
            if "Scale." + name in saved_state_dict and param.size() == saved_state_dict["Scale." + name].size():
                new_params[name].copy_(saved_state_dict["Scale." + name])
        else:
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    # init ema-model
    if train_unlabeled:
        ema_model = create_ema_model(model)
        ema_model.train()
        ema_model = ema_model.cuda()
    else:
        ema_model = None

    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    if dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = Compose([RandomCrop_city(input_size)])
        else:
            data_aug = None

        # Use img_size=input_size=(512,512) as random cropping is applied before resizing (only for cityscapes loader)
        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)

    train_dataset_size = len(train_dataset)
    print ('dataset size: ', train_dataset_size)

    if labeled_samples is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)

        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)
        trainloader_remain_iter = iter(trainloader_remain)

    else:
        partial_size = labeled_samples
        print('Training on number of samples:', partial_size)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)

        trainloader_remain_iter = iter(trainloader_remain)


    #New loader for Domain transfer
    if True:
        data_loader = get_loader('synthia')
        data_path = get_data_path('synthia')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None

        # Use img_size=(1280,720) as random cropping is applied after resizing (only for gta loader)
        train_dataset = data_loader(data_path, list_path = './data/synthia_list/train.txt', augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)

    trainloader_iter = iter(trainloader)
    print('synthia size:',len(trainloader))

    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    accumulated_loss_l_S = []
    accumulated_loss_l_E = []
    accumulated_loss_u = []
    accumulated_loss_u_S = []
    accumulated_loss_u_E = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    epochs_since_start = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_u_S_value = 0
        loss_u_E_value = 0
        loss_l_value = 0
        loss_l_S_value = 0
        loss_l_E_value = 0
        
        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)


        weak_parameters={"flip": 0}
        images, labels, _, edge = batch
        images = images.cuda()
        edge = edge.cuda()
        labels = labels.cuda().long()

        out = model(images)
        pred = interp(out["S"])
        pred_init = interp(out["initial_S"])
        predE = torch.sigmoid(interp(out["D_src"])).squeeze()
        predE_init = torch.sigmoid(interp(out["initial_D_src"])).squeeze()
        
        L_E_l = loss_edge_calc(predE_init, edge) + loss_edge_calc(predE, edge)
        L_S_l = loss_calc(pred, labels) + loss_calc(pred_init, labels)
        L_l = L_S_l + config['training']['edge_loss_weight'] * L_E_l # Cross entropy loss for labeled data

        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            
            images_remain, _, _, _, edge_remain = batch_remain
            images_remain = images_remain.cuda()
            edge_remain = edge_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data = images_remain)
            ## Modified
            out_tar = ema_model(inputs_u_w) 
            logits_u_w = interp(out_tar["S"])
            logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data = logits_u_w.detach())
            
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            if mix_mask == "class":
                for image_i in range(batch_size):
                    classes = torch.unique(labels[image_i])
                    nclasses = classes.shape[0]
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()

                    if image_i == 0:
                        MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()

            elif mix_mask == None:
                MixMask = torch.ones((inputs_u_w.shape))

            strong_parameters = {"Mix": MixMask0}
            if random_flip:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if color_jitter:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if gaussian_blur:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0

            inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat((images[0].unsqueeze(0),images_remain[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            inputs_u_s1, _ = strongTransform(strong_parameters, data = torch.cat((images[1].unsqueeze(0),images_remain[1].unsqueeze(0))))
            inputs_u_s = torch.cat((inputs_u_s0,inputs_u_s1))
            out_u_s = model(inputs_u_s)
            logits_u_s = interp(out_u_s["S"])
            logits_u_s_init = interp(out_u_s["initial_S"])

            strong_parameters["Mix"] = MixMask0
            _, targets_u0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),targets_u_w[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, targets_u1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),targets_u_w[1].unsqueeze(0))))
            targets_u = torch.cat((targets_u0,targets_u1)).long()

            strong_parameters["Mix"] = MixMask0
            _, edge_u0 = strongTransform(strong_parameters, target = torch.cat((edge[0].unsqueeze(0),edge_remain[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, edge_u1 = strongTransform(strong_parameters, target = torch.cat((edge[1].unsqueeze(0),edge_remain[1].unsqueeze(0))))
            edge_u = torch.cat((edge_u0,edge_u1))

            edge_pred_u = interp(out_u_s["D_src"])
            edge_pred_ui = interp(out_u_s["initial_D_src"])

            
            unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
            pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
            
            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()

            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                L_u = consistency_weight * unlabeled_loss(logits_u_s, targets_u, pixelWiseWeight)
                L_u2 = consistency_weight * unlabeled_loss(logits_u_s_init, targets_u, pixelWiseWeight)
                
                edge_pred_u = torch.sigmoid(edge_pred_u).squeeze()
                edge__pred_ui = torch.sigmoid(edge_pred_ui).squeeze()
                

                L_E =  loss_edge_calc(edge_pred_u, edge_u.detach())
                L_E2 = loss_edge_calc(edge_pred_ui, edge_u.detach())
                L_S_u = L_u + L_u2
                L_E_u = L_E + L_E2
                L_u = L_S_u + config['training']['edge_loss_weight'] * L_E_u
                
            loss = L_l + L_u

        else:
            loss = L_l

        if len(gpus) > 1:
            loss = loss.mean()
            loss_l_value += L_l.mean().item()
            loss_l_S_value += L_S_l.mean().item()
            loss_l_E_value += L_E_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.mean().item()
                loss_u_S_value += L_S_u.mean().item()
                loss_u_E_value += L_E_u.mean().item()
        else:
            loss_l_value += L_l.item()
            loss_l_S_value += L_S_l.mean().item()
            loss_l_E_value += L_E_l.mean().item()
            if train_unlabeled:
                loss_u_value += L_u.item()
                loss_u_S_value += L_S_u.mean().item()
                loss_u_E_value += L_E_u.mean().item()

        loss.backward()
        optimizer.step()

        # update Mean teacher network
        if ema_model is not None:
            alpha_teacher = config['training']['teacher_alpha'] 
            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=i_iter)

        print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(i_iter, num_iterations, loss_l_value, loss_u_value))

        if i_iter % save_checkpoint_every == 0 and i_iter!=0:
            if epochs_since_start * len(trainloader) < save_checkpoint_every:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            else:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            accumulated_loss_l_S.append(loss_l_S_value)
            accumulated_loss_l_E.append(loss_l_E_value)
            if train_unlabeled:
                accumulated_loss_u.append(loss_u_value)
                accumulated_loss_u_S.append(loss_u_S_value)
                accumulated_loss_u_E.append(loss_u_E_value)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                tensorboard_writer.add_scalar('Training/Supervised seg loss', np.mean(accumulated_loss_l_S), i_iter)
                tensorboard_writer.add_scalar('Training/Supervised edge loss', np.mean(accumulated_loss_l_E), i_iter)
                accumulated_loss_l = []
                accumulated_loss_l_S = []
                accumulated_loss_l_E = []
                
                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                    tensorboard_writer.add_scalar('Training/Unsupervised seg loss', np.mean(accumulated_loss_u_S), i_iter)
                    tensorboard_writer.add_scalar('Training/Unsupervised edge loss', np.mean(accumulated_loss_u_E), i_iter)
                    accumulated_loss_u = []
                    accumulated_loss_u_S = []
                    accumulated_loss_u_E = []

        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            if dataset == 'cityscapes':
                mIoU, cIoU, eval_loss = evaluate(model, dataset, config, save_dir=checkpoint_dir)
            model.train()

            if mIoU > best_mIoU and save_best_model:
                best_mIoU = mIoU
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

            if config['utils']['tensorboard']:
                tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
                tensorboard_writer.add_scalar('Validation/Loss', eval_loss, i_iter)
                for i, iou in enumerate(cIoU):
                    tensorboard_writer.add_scalar(f'Validation/xIoU_{i}_{cityscapesLoader.class_names[i + 1]}', iou, i_iter)

    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)

    model.eval()
    if dataset == 'cityscapes':
        mIoU, cIoU, val_loss = evaluate(model, dataset, config, save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)
    if num_iterations == 200000:
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True, schedule=True)

    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)
        for i, iou in enumerate(cIoU):
            tensorboard_writer.add_scalar(f'Validation/xIoU_{i}_{cityscapesLoader.class_names[i + 1]}', iou, i_iter)


    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')

if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
    elif config['pretrained'] == 'imagenet':
        print("loaded imagenet weight")
        restore_from = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

    num_classes=19
    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']
    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)
    ignore_label = config['ignore_label'] 
    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']
    setup_seeds(random_seed)
    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']
    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']
    save_best_model = config['utils']['save_best_model']
    save_unlabeled_images = args.save_images or config['utils'].get('save_unlabeled_imgs', False)

    gpus = (0,1,2,3)[:args.gpus]

    main()

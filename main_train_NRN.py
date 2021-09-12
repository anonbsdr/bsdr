import argparse
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import random, string
import math
import pickle
from collections import OrderedDict
import torch
from torch import nn as nn, optim as optim
from torch.autograd import Variable

import datetime

from pdb import set_trace as bp
from models import NRN_downscale_factor
from models import BSDR_downscale_factor
from models import NRN

import warnings 
warnings.filterwarnings("ignore")
np.set_printoptions(precision=5)
import time
import util

parser = argparse.ArgumentParser(description='PyTorch NRN Training')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU number')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4),only used for train')
parser.add_argument('--patches', default=1, type=int, metavar='N',
                    help='number of patches per image')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--count-thresh', default='1250', type=str, help="Define the count threshold for categories")

networks = []
config = {
    'max_count':4500,
    'fulltopatch_countratio' : 5.0,
    'image_patch_size' : 256,
    'noisy_dist_param' : np.array([[4.0,4.0],[16.0,4.0]])
}

def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')


# Get the filename for the model stored after 'epochs_over' epochs got over
def get_filename(net_name, epochs_over):
    return net_name + "_epoch_" + str(epochs_over) + ".pth"


def save_checkpoint(state, fdir, name='checkpoint.pth'):
    filepath = os.path.join(fdir, name)
    torch.save(state, filepath)


def train_function(Xs, Ys, network, optimizer):
    # torch.cuda.empty_cache()
    network = network.cuda()
    optimizer.zero_grad()
    network.train()
    X = torch.autograd.Variable(torch.from_numpy(Xs)).cuda()
    Y = torch.autograd.Variable(torch.FloatTensor(Ys)).cuda()
    X.requires_grad = False
    Y.requires_grad = False
    outputs = network(X)
    upsample = nn.Upsample(scale_factor = NRN_downscale_factor, mode='nearest')

    loss = 0.0
    loss_criterion = nn.MSELoss(size_average=True)

    assert(upsample(outputs).shape == X.shape == Y.shape)
    outputs = upsample(outputs) * X

    loss = loss_criterion(outputs,Y)
    assert(loss.grad_fn != None)
    loss.backward()
    optimizer.step()
    return loss.item()


def check_conv_gradient_change(network):

    s = []
    count = 0

    for _,main_module in network.named_children():
        for name, module in main_module.named_children():
            if isinstance(module, nn.Conv2d):
                s.append(module.weight.data.cpu().detach().numpy().reshape(-1))
                count += 1

    return np.concatenate(s)

    

def construct_random_GT(current_density_range, dmap_shape, num_grids = 2):
    # num_grids = 2
    num_cells = num_grids * num_grids
    cell_h,cell_w = dmap_shape[0]/num_grids,dmap_shape[1]/num_grids
    range_min = current_density_range.min()+1
    range_max = current_density_range.max()
    cell_counts = np.random.uniform(range_min,range_max, num_cells) / num_cells
    cell_counts = np.floor(cell_counts).astype('int').reshape((num_grids,num_grids))
    if cell_counts.sum() < range_min:
        cell_counts[0,0] += abs(cell_counts.sum() - range_min)
    all_locs = []
    for row_idx in range(num_grids):
        for col_idx in range(num_grids):
            loc_y = np.random.randint(0,cell_h,cell_counts[row_idx,col_idx]) + row_idx * cell_h
            loc_x = np.random.randint(0,cell_w,cell_counts[row_idx,col_idx]) + col_idx * cell_w
            locs = np.hstack((loc_x[:,None],loc_y[:,None])) #(N,2)
            all_locs.append(locs)
    all_locs = np.concatenate(all_locs)
    assert(all_locs.shape == (cell_counts.sum(),2))
    hmap = util._create_heatmap(dmap_shape, dmap_shape, all_locs, density_map_kernel)
    # bp()
    assert(abs(hmap.sum() - cell_counts.sum()) < 0.5)
    assert(hmap.shape == dmap_shape)
    return hmap

def train_network():
    global networks
    for i in range(num_density_categories):
        networks.append(NRN(name='NRN_'+str(i)))

    assert(len(networks) == num_density_categories)
    
    # load_model_VGG16(network)
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'snapshots'))

    global f
    snapshot_path = os.path.join(model_save_path, 'snapshots')
    f = open(os.path.join(model_save_path, 'train0.log'), 'w')

    # -- Logging Parameters
    log(f, 'args: ' + str(args))
    log(f,'Number of networks {}'.format(len(networks)))
    log(f, 'model: ' + str(networks[0]), False)

    log(f, 'Training0...')
    log(f, 'LR: %.12f.' % (args.lr))
    log(f, 'config {}'.format(config))
    start_epoch = 0
    num_epochs = args.epochs

    train_losses = {}

    for metric in ['loss1']:
        train_losses[metric] = []

    batch_size = args.batch_size
    num_train_images = 240
    num_patches_per_image = args.patches
    assert(batch_size < (num_patches_per_image * num_train_images))
    num_batches_per_epoch = num_patches_per_image * num_train_images // batch_size
    assert(num_batches_per_epoch >= 1)
    log(f,'num_batches_per_epoch {}'.format(num_batches_per_epoch))
    # bp()
    optimizers = []
    for i in range(len(networks)):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, networks[i].parameters()),
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizers.append(optimizer)
    # -- Main Training Loop

    log(f,'count_density_threshold {} '.format(count_density_threshold))
    log(f, 'noisy_dist_param {}'.format(noisy_dist_param))
    assert(noisy_dist_param.shape[0] == len(networks))
    assert(len(networks) == 2)

    
    all_train_epochs_loss = []

    Ys_map_shape = (image_crop_size//BSDR_downscale_factor, image_crop_size//BSDR_downscale_factor)
    for e_i, epoch in enumerate(range(start_epoch, num_epochs)):
        avg_loss = []

        for b_i in range(num_batches_per_epoch):
            
            global blur_sigma
            all_net_train_loss = []
            before_conv_sum = []
            for network_idx in range(len(networks)):

                current_range = count_density_range[network_idx:network_idx+2]

                sample_Y_maps = []
                for i in range(batch_size):
                    hmap = construct_random_GT(current_range,Ys_map_shape)
                    hmap = hmap[None,...]
                    sample_Y_maps.append(hmap)
                sample_Y_maps = np.array(sample_Y_maps)
                assert(sample_Y_maps.shape == (batch_size,1,)+Ys_map_shape)
                sample_Ys_counts = sample_Y_maps.reshape((sample_Y_maps.shape[0],-1)).sum(axis=1).astype('int')
                sample_count_idx = np.digitize(sample_Ys_counts,count_density_threshold,right=True)
                assert(np.all(sample_count_idx == network_idx))
                
                noisy_counts = np.random.normal(noisy_dist_param[network_idx, 0], noisy_dist_param[network_idx, 1], batch_size)
                new_noisy_maps = sample_Y_maps.copy().astype('float32')
                new_noisy_maps = new_noisy_maps / (np.sum(new_noisy_maps ,axis=(1,2,3)))[:,None,None,None]
                new_noisy_maps = new_noisy_maps * noisy_counts[:,None,None,None]
                mask = np.isnan(np.sum(new_noisy_maps ,axis=(1,2,3)))
                new_noisy_maps[mask] = sample_Y_maps[mask]
                # bp()
                new_noisy_maps = new_noisy_maps.astype('float32')
                assert(sample_Y_maps.shape  == new_noisy_maps.shape)

                before_conv_sum.append(check_conv_gradient_change(networks[network_idx]))
                train_loss = train_function(new_noisy_maps, sample_Y_maps, networks[network_idx], optimizers[network_idx])#sampled_GT
                all_net_train_loss.append(train_loss)
            avg_loss.append(all_net_train_loss)

            # Logging losses after 1k iterations.
            if b_i % 5 == 0:
                # bp()
                log(f, 'Epoch %d [%d]: %s loss: %s.' % (epoch, b_i, [networks[i].name for i in range(len(networks))], np.array(all_net_train_loss).round(5)))
            after_conv_sum = []
            for i in range(len(networks)):
                after_conv_sum.append(check_conv_gradient_change(networks[i]))

            for i in range(len(networks)):
                assert(np.any(before_conv_sum[i]!=after_conv_sum[i]))

        avg_loss = np.mean(np.array(avg_loss),axis=0)
        assert(avg_loss.shape == (len(networks),))
        train_losses['loss1'].append(avg_loss)
        log(f, 'TRAIN NRN epoch: ' + str(epoch) + ' train mean loss1:' + str(avg_loss))
        all_train_epochs_loss.append(avg_loss)
        log(f,'Best train epochs by loss {}'.format(np.argmin(np.array(all_train_epochs_loss),axis=0)))
        torch.cuda.empty_cache()
        
        # Save networks
        for i in range(len(networks)):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': networks[i].state_dict(),
                'optimizer': optimizer.state_dict(),
            }, snapshot_path, get_filename(networks[i].name, epoch + 1))

        print('saving graphs...')
        with open(os.path.join(snapshot_path, 'losses.pkl'), 'wb') as lossfile:
            pickle.dump((train_losses), lossfile, protocol=2)

        for metric in train_losses.keys():
                # print(metric, "METRIC", train_losses[metric])
            all_net_losses = np.array(train_losses[metric])
            for i in range(len(networks)):
                plt.plot(all_net_losses[:,i])
            plt.savefig(os.path.join(snapshot_path, 'train_%s.png' % metric))
            plt.clf()
            plt.close()

    min_train_epochs = np.argmin(np.array(all_train_epochs_loss),axis=0)
    nrn_models_list = ['NRN_{}_epoch_{}.pth'.format(i,min_train_epochs[i]+1) for i in range(num_density_categories)]
    with open(os.path.join(snapshot_path, 'best_model.pkl'), 'wb') as file:
        pickle.dump(nrn_models_list , file, protocol=2)
    log(f, 'Exiting train...')
    f.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # -- Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # -- Assertions
    np.random.seed(11)
    random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    model_save_dir = './models_NRN'

    noisy_dist_param = config['noisy_dist_param']
    
    blur_sigma = 1
    density_map_kernel = util._gaussian_kernel(sigma = blur_sigma)

    batch_size = args.batch_size
    count_density_threshold = str(args.count_thresh).split(',') if args.count_thresh !='' else []
    count_density_threshold = list(map(lambda x: float(x),count_density_threshold))

    image_crop_size = int(config['image_patch_size'])
    fulltopatch_countratio = float(config['fulltopatch_countratio'])
    count_density_threshold = [i/fulltopatch_countratio for i in count_density_threshold]

    max_gt_count = config['max_count']/fulltopatch_countratio
    count_density_range = np.concatenate(([0],count_density_threshold,[max_gt_count]))

    num_density_categories = len(count_density_threshold) + 1
    assert(noisy_dist_param.shape[0] == num_density_categories)
    # -- Train the model
    train_network()

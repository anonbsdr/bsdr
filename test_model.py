import argparse
import random
from crowd_dataset import CrowdDataset
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
import scipy.stats as ss

from pdb import set_trace as bp
from models import BSDR_Net
from models import load_rot_model_blocks, check_BN_no_gradient_change
from models import check_conv_no_gradient_change, set_batch_norm_to_eval
from models import load_net
from noisy_gts import create_noisy_gt
from models import NRN
import warnings 
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch BSDR Testing')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU number')
parser.add_argument('--dataset', default="parta", type=str,
                    help='dataset to train on')
parser.add_argument('--model-name', default="", type=str,
                    help='name of model file')

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


def print_graph(maps, title, save_path):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, (map, args) in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        if len(map.shape) > 2 and map.shape[0] == 3:
            # bp()
            plt.imshow(map.transpose((1, 2, 0)).astype(np.uint8),aspect='equal', **args)
            # bp()
        else:
            # bp()
            
            plt.imshow(map, aspect='equal', **args)
            plt.colorbar()
            # bp()
            plt.axis('off')
    plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches = 0)
    fig.clf()
    plt.clf()
    plt.close()

excluded_layers = ['conv4_1', 'conv4_2', 'conv5_1']


@torch.no_grad()
def test_function(X, Y, network):
    """
    Evaluation of network on test and valid set
    Parameters
    ----------
    X : input images (B,3,h,w)
    Y : ground truth (B,1,h/8,w/8)
    network : BSDR object
    """
    X = torch.autograd.Variable(torch.from_numpy(X)).cuda()
    Y = torch.autograd.Variable(torch.from_numpy(Y)).cuda()

    network = network.cuda()
    network.eval()
    output = network(X) # (B,1,h,w)
    loss = 0.0
    loss_criterion = nn.MSELoss(size_average=True)
    # bp()

    loss = loss_criterion(output, Y)
    
    count_error = torch.abs(torch.sum(Y.view(Y.size(0), -1), dim=1) - torch.sum(output.view(output.size(0), -1), dim=1))
    network.train()
    network = set_batch_norm_to_eval(network)
    return loss.item(), output.cpu().detach().numpy(), count_error.cpu().detach().numpy()


def test_network(dataset, set_name, network, print_output=False):
    """
    Main loop for evaluation of BSDR network
    Parameters
    ----------
    dataset : dataset object for retrieving data from test/valid set
    set-name : choose the test / valid set
    network : BSDR object
    print_output : determine to dump predictions
    """
    if isinstance(print_output, str):
        print_path = print_output
    elif isinstance(print_output, bool) and print_output:
        print_path = model_save_dir+'/dump'
    else:
        print_path = None

    loss_list = []
    count_error_list = []
    for idx, data in enumerate(dataset.test_get_data(set_name)):
        image_name, Xs, Ys = data
        image = Xs[0].transpose((1, 2, 0))
        image = cv2.resize(image, (image.shape[1] // output_downscale, image.shape[0] // output_downscale))

        loss, pred_dmap, count_error = test_function(Xs, Ys, network)
        # bp()
        max_val = max(np.max(pred_dmap[0, 0].reshape(-1)), np.max(Ys[0, 0].reshape(-1)))
        maps = [(np.transpose(image,(2,0,1)), {}),
                (pred_dmap[0,0], {'cmap': 'jet', 'vmin': 0., 'vmax': max_val}),
                (Ys[0,0], {'cmap': 'jet', 'vmin': 0., 'vmax': max_val})]
        # bp()
        loss_list.append(loss)
        count_error_list.append(count_error)

        # -- Plotting boxes
        if print_path:
            print_graph(maps, "Gt:{},Pred:{}".format(np.sum(Ys),np.sum(pred_dmap)), os.path.join(print_path, image_name))


    loss = np.mean(loss_list)
    mae = np.mean(count_error_list)
    mse = np.sqrt(np.mean(np.square(count_error_list)))
    return {'loss1':loss,'new_mae':mae,'mse':mse}, mae

def train_network():
    """
    Main training loop for BSDR
    """
    network = BSDR_Net()
    
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'snapshots'))
        os.makedirs(os.path.join(model_save_dir,'dump'))
        os.makedirs(os.path.join(model_save_dir,'dump_test'))
    global f
    snapshot_path = os.path.join(model_save_path, 'snapshots')
    f = open(os.path.join(model_save_path, 'train0.log'), 'w')

    # -- Logging Parameters
    log(f, 'args: ' + str(args))
    log(f, 'model: ' + str(network), False)

    network = load_net(network,'models_BSDR/train2/snapshots',str(args.model_name))

    log(f, 'Testing...')
    epoch_test_losses, mae = test_network(dataset, 'test', network, False)
    log(f, 'TEST epoch: ' + str(-1) + ' test loss1, mae:' + str(epoch_test_losses))

    return


if __name__ == '__main__':
    args = parser.parse_args()
    # -- Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # -- Assertions
    assert (args.dataset)

    # -- Setting seeds for reproducability
    np.random.seed(11)
    random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)

    # -- Dataset paths
    if args.dataset == "parta":
        validation_set = 60
        path = '../dataset/ST_partA/'
        output_downscale = 8
        density_map_sigma = 1
        blur_sigma = 1
        image_size_min = 256
        image_crop_size = 256
        network_output_downscale = 4
    elif args.dataset == "ucfqnrf":
        validation_set = 240
        output_downscale = 8
        path = '../dataset/UCF-QNRF_ECCV18/'
        output_downscale = 8
        density_map_sigma = 1
        blur_sigma = 1
        image_size_min = 256
        image_crop_size = 256
        network_output_downscale = 4
    else:
        validation_set = 0
        output_downscale = 8
        path = '../../dataset/ST_partA_' + args.dataset.replace('parta_', '') + '/'

    model_save_dir = './models_BSDR_test'

    dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale,density_map_sigma=density_map_sigma,
                           image_size_multiple = output_downscale * network_output_downscale,
                           image_size_min = image_size_min , image_crop_size = image_crop_size)
    #print(dataset.data_files['test_valid'], len(dataset.data_files['test_valid']))
    print(dataset.data_files['train'], len(dataset.data_files['train']))

    # -- Train the model
    train_network()

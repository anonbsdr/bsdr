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


parser = argparse.ArgumentParser(description='PyTorch BSDR Training')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU number')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4),only used for train')
parser.add_argument('--patches', default=1, type=int, metavar='N',
                    help='number of patches per image')
parser.add_argument('--dataset', default="parta", type=str,
                    help='dataset to train on')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument("--use-noisygt", type=bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
parser.add_argument('--count-thresh', default='1250', type=str, help="Define the count threshold for categories")


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

def train_function(Xs, Ys, network, optimizer):
    """
    Training Prediction and Loss computation
    Parameters
    ----------
    Xs : input images (B,3,h,w)
    Ys : ground truth (B,1,h/8,w/8)
    network : BSDR object
    optimizer : training optimizer
    """

    network = network.cuda()
    optimizer.zero_grad()

    X = torch.autograd.Variable(torch.from_numpy(Xs)).cuda()
    Y = torch.autograd.Variable(torch.FloatTensor(Ys)).cuda()

    outputs = network(X)

    loss = 0.0
    loss_criterion = nn.MSELoss(size_average=True)
    loss =  loss_criterion(outputs, Y)

    assert(loss.grad_fn != None)
    loss.backward()
    optimizer.step()
    return loss.item()


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

def check_conv_gradient_change(network):
    """
        Check if conv layers are frozen
        Parameters
        ----------
        network : NRN network object
        """
    s = []
    count = 0
    for _,main_module in network.named_children():
        for name, module in main_module.named_children():
            if isinstance(module, nn.Conv2d):
                s.append(module.weight.data.cpu().detach().numpy().reshape(-1))
                count += 1
    return np.concatenate(s)

def train_network():
    """
    Main training loop for BSDR
    """
    network = BSDR_Net()
    nrn_networks = []
    nrn_snapshot_path = 'models_NRN/train2/snapshots'
    nrn_models_list = pickle.load(open(os.path.join(nrn_snapshot_path,'best_model.pkl'),'rb'))

    before_nrn_sum = []
    # bp()
    for i in range(num_density_categories):
        nrn_net = NRN()
        nrn_model_file = nrn_models_list[i]
        nrn_net = load_net(nrn_net,nrn_snapshot_path,nrn_model_file)
        nrn_net = nrn_net.cuda()
        nrn_net.eval()
        nrn_networks.append(nrn_net)
        before_nrn_sum.append(check_conv_gradient_change(nrn_net))

    # load_model_VGG16(network)
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
    log(f, 'Training0...')
    log(f, 'LR: %.12f.' % (args.lr))
    log(f,'NRN folder {}'.format(nrn_snapshot_path))
    log(f,'NRN models : {}'.format(nrn_models_list))

    start_epoch = 0
    num_epochs = args.epochs
    valid_losses = {}

    train_losses = {}
    for metric in ['loss1', 'new_mae','mse']:
        valid_losses[metric] = []

    for metric in ['loss1']:
        train_losses[metric] = []

    batch_size = args.batch_size
    num_train_images = len(dataset.data_files['train'])
    num_patches_per_image = args.patches
    assert(batch_size < (num_patches_per_image * num_train_images))
    num_batches_per_epoch = num_patches_per_image * num_train_images // batch_size
    assert(num_batches_per_epoch >= 1)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    network = load_rot_model_blocks(network, snapshot_path='models_rot_net/train2/snapshots', excluded_layers=excluded_layers)

    # -- Main Training Loop
    all_epoch_test_accs = []

    global sampled_GT

    log(f, 'Testing...')
    epoch_test_losses, mae = test_network(dataset, 'test', network, False)
    log(f, 'TEST epoch: ' + str(-1) + ' test loss1, mae:' + str(epoch_test_losses))


    for e_i, epoch in enumerate(range(start_epoch, num_epochs)):
        avg_loss = []

        global blur_sigma
        for b_i in range(num_batches_per_epoch):
            # Generate next training sample
            Xs, Ys, Ys_full_counts = dataset.train_get_data(batch_size=args.batch_size)
            category_labels = np.digitize(Ys_full_counts, count_density_threshold,right = True)
            if args.use_noisygt:

                before_noisy_gt_maps = []
                
                for i in range(batch_size):
                    image = Xs[i].transpose((1,2,0)).astype('uint8')#(224,224,3)
                    noisy_gt_map = create_noisy_gt(image,output_downscale,blur_sigma)
                    before_noisy_gt_maps.append(noisy_gt_map[None,...])    
                before_noisy_gt_maps = np.array(before_noisy_gt_maps)
                # bp()
                assert(Ys.shape == before_noisy_gt_maps.shape)
                before_noisy_X = torch.autograd.Variable(torch.from_numpy(before_noisy_gt_maps)).cuda().float()
                before_noisy_X.requires_grad = False
                Ys_counts = Ys.reshape((Ys.shape[0],-1)).sum(axis=1).astype('int')
                assert(Ys_counts.shape == Ys_full_counts.shape)

                factor_arr = []
                upsample = nn.Upsample(scale_factor=network_output_downscale, mode='nearest')
                for i in range(batch_size):
                    factor = nrn_networks[category_labels[i]](before_noisy_X[i][None,...])[0]
                    factor_arr.append(factor.detach().cpu().numpy())

                factor_arr = np.array(factor_arr)
                factor_arr = torch.autograd.Variable(torch.from_numpy(factor_arr)).cuda().float()
                # bp()
                after_noisy_X = upsample(factor_arr) * before_noisy_X
                after_noisy_gt_maps = after_noisy_X.detach().cpu().numpy()
                assert(Ys.shape == after_noisy_gt_maps.shape == before_noisy_gt_maps.shape)
                Ys = after_noisy_gt_maps

            train_loss = train_function(Xs, Ys, network, optimizer)#sampled_GT
            avg_loss.append(train_loss)
            for i in range(num_density_categories):
                after_sum = check_conv_gradient_change(nrn_networks[i])
                assert(np.all(before_nrn_sum[i] == after_sum))
            # Logging losses after 1k iterations.
            if b_i % 10 == 0:
                log(f, 'Epoch %d [%d]: %s loss: %s.' % (epoch, b_i, [network.name], train_loss))

        avg_loss = np.mean(np.array(avg_loss))
        train_losses['loss1'].append(avg_loss)
        log(f, 'TRAIN epoch: ' + str(epoch) + ' train mean loss1:' + str(avg_loss))

        torch.cuda.empty_cache()
        log(f, 'Testing...')

        epoch_val_losses, valid_mae = test_network(dataset, 'test_valid', network, False)
        log(f, 'TEST valid epoch: ' + str(epoch) + ' test valid loss1, mae' + str(epoch_val_losses))
        # exit(0)

        for metric in ['loss1', 'new_mae','mse']:
            valid_losses[metric].append(epoch_val_losses[metric])

        min_valid_epoch = np.argmin(valid_losses['new_mae'])

        log(f,'Best valid so far epoch: {}, valid mae: {}'.format(
            min_valid_epoch,
            valid_losses['new_mae'][min_valid_epoch]))
        # Save networks
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, snapshot_path, get_filename(network.name, epoch + 1))

        print('saving graphs...')
        with open(os.path.join(snapshot_path, 'losses.pkl'), 'wb') as lossfile:
            pickle.dump((train_losses, valid_losses), lossfile, protocol=2)

        for metric in train_losses.keys():
            if "maxima_split" not in metric:
                if isinstance(train_losses[metric][0], list):
                    for i in range(len(train_losses[metric][0])):
                        plt.plot([a[i] for a in train_losses[metric]])
                        plt.savefig(os.path.join(snapshot_path, 'train_%s_%d.png' % (metric, i)))
                        plt.clf()
                        plt.close()
                # print(metric, "METRIC", train_losses[metric])
                plt.plot(train_losses[metric])
                plt.savefig(os.path.join(snapshot_path, 'train_%s.png' % metric))
                plt.clf()
                plt.close()

        for metric in valid_losses.keys():
            if isinstance(valid_losses[metric][0], list):
                for i in range(len(valid_losses[metric][0])):
                    plt.plot([a[i] for a in valid_losses[metric]])
                    plt.savefig(os.path.join(snapshot_path, 'valid_%s_%d.png' % (metric, i)))
                    plt.clf()
                    plt.close()
            plt.plot(valid_losses[metric])
            plt.savefig(os.path.join(snapshot_path, 'valid_%s.png' % metric))
            plt.clf()
            plt.close()


    min_valid_epoch = np.argmin(valid_losses['new_mae'])
    network = load_net(network,snapshot_path,get_filename(network.name, min_valid_epoch + 1))
    log(f,'Testing on best model {}'.format(min_valid_epoch))
    epoch_test_losses, mae = test_network(dataset, 'test', network, print_output=os.path.join(model_save_dir,'dump_test'))
    log(f, 'TEST epoch: ' + str(epoch) + ' test loss1, mae:' + str(epoch_test_losses))
    log(f, 'Exiting train...')
    f.close()
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

    model_save_dir = './models_BSDR'

    batch_size = args.batch_size
    count_density_threshold = str(args.count_thresh).split(',') if args.count_thresh !='' else []
    count_density_threshold = list(map(lambda x: float(x),count_density_threshold))

    num_density_categories = len(count_density_threshold) + 1
    dataset = CrowdDataset(path, name=args.dataset, valid_set_size=validation_set,
                           gt_downscale_factor=output_downscale,density_map_sigma=density_map_sigma,
                           image_size_multiple = output_downscale * network_output_downscale,
                           image_size_min = image_size_min , image_crop_size = image_crop_size)
    #print(dataset.data_files['test_valid'], len(dataset.data_files['test_valid']))
    print(dataset.data_files['train'], len(dataset.data_files['train']))

    # -- Train the model
    train_network()

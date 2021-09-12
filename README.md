# Dense or Sparse: Crowd Counting using Binary Supervision

# Dataset Requirements
Download Shanghaitech dataset from [here](https://github.com/desenzhou/ShanghaiTechDataset).
Download UCF-QNRF dataset from [here](http://crcv.ucf.edu/data/ucf-qnrf/).

Place the dataset in `../dataset/` folder. (`dataset` and `bsdr` folders should have the same parent directory). So the directory structure should look like the following:

```
-- bsdr
   -- models.py
   -- main_train_NRN.py
   -- ....
-- dataset
   --ST_partA
     -- test_data
      -- ground-truth
      -- images
     -- train_data
      -- ground-truth
      -- images
  --UCF-QNRF
    --Train
      -- ...
    --Test
      -- ...
```

# Dependencies and Installation
We strongly recommend to run the codes in Nvidia-Docker. Install both `docker` and `nvidia-docker` (please find instructions from their respective installation pages).
After the docker installations, pull pytorch docker image with the following command:
`docker pull nvcr.io/nvidia/pytorch:18.04-py3`
and run the image using the command:
`nvidia-docker run --rm -ti --ipc=host nvcr.io/nvidia/pytorch:18.04-py3`

Further software requirements are listed in `requirements.txt`. 

To install them type, `pip install -r requirements.txt`

The code has been run and tested on `Python 3.6.4`, `Cuda 9.0, V9.0.176` and `PyTorch 0.4.1`. 

# Usage

## Pretrained Models

The pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1ND1I3d2r0ny5YRWYc4GMQKifntXCDPJg?usp=sharing). The directory structure is as follows:

```
-- parta_models
   -- models_rot_net
     -- best_model.pkl
     -- rot_net_parta.pth
   -- models_NRN
     -- NRN_sparse_parta.pth
     -- NRN_dense_parta.pth
     -- best_model.pkl
   -- models_BSDR
     -- BSDR_parta.pth
     
-- ucfqnrf_models
   -- models_rot_net
     -- ...
   -- models_NRN
     -- ...
   -- models_BSDR
     -- ...
```

* For testing the BSDR pretrained models, save the pretrained weights files from `{dataset}/models_BSDR` in `models_BSDR/train2/snapshots/` and follow the steps outlined in Testing section.

* For training only BSDR using self supervision pretrained model and NRN pretrained model, save the pretrained weights files from `{dataset}/models_rot_net` in `models_rot_net/train2/snapshots/` , and files from `{dataset}/models_NRN` in `models_NRN/train2/snapshots/` and follow steps for BSDR training.

## Testing

After either finishing the training or downloading pretrained models, the model can be tested using the below script.
The model must be present in `models_BSDR/train2/snapshots`.

* `python test_model.py --dataset parta --gpu=0 --model-name BSDR_parta.pth `
```
--dataset = parta / ucfqnrf
--model-name = Name of the model checkpoint to be tested
```

## Training
After downloading the datasets and installing all dependencies, proceed with training as follows:

### Self Supervision Rotation Task Training:
* `python main_train_RotNet.py --dataset parta --gpu 0`
```
  -b = Batch size [For ucfqnrf, set 16]
  --dataset = parta / ucfqnrf
  --gpu = GPU Number
  --epochs = Number of epochs to train
```
### NRN Training:
* `python main_train_NRN.py --dataset parta --gpu 0 --count-thresh=1250`
```
  --dataset = parta / ucfqnrf
  --gpu = GPU Number
  --count-thresh = count density threshold to categorise the training images as sparse or dense
```
### BSDR Training:
* `python main_train_BSDR.py --dataset parta --gpu 0 --count-thresh=1250 --use-noisygt --epochs=500`
```
  --dataset = parta / ucfqnrf
  --gpu = GPU Number
  --count-thresh = count density threshold to categorise the training images as sparse or dense
  --use-noisygt = if True the rectified density map is the target map or else the actual ground truth density map
  --epochs = number of training epochs [For ucfqnrf , set 100]
```

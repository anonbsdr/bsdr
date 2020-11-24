"""
crowd_dataset.py: Code to use crowd counting datasets for training and testing.
"""

import numpy as np
import cv2
import os
import random
import scipy.io
import pickle
from pdb import set_trace as bp
import util
class CrowdDataset:
    """
    Class to use crowd counting datasets for training and testing.
    Version: 3.1
    DataReader supports the following:
        ground truths: can create dot maps & pseudo box maps.
        testing: full image testing.
        training: extract random crops with flip augmentation.
    """

    def __init__(self, data_path, name='parta', valid_set_size=0,
                 gt_downscale_factor=2,
                 image_size_min=224, image_size_max=1024, image_crop_size=224,
                 image_size_multiple = -1,
                 density_map_sigma=1.0):
        """
        Initialize dataset class.

        Parameters
        ----------
        data_path: string
            Path to the dataset data; the provided directory MUST have following structure:
                |-train_data
                  |-images
                  |-ground_truth
                |-test_data
                  |-images
                  |-ground_truth
        name: string
            Dataset name; MUST be one of ['parta', 'partb',
                                          'ucfqnrf', 'ucf50']
        valid_set_size: int
            Number of images from train set to be randomly taken for validation. Value MUST BE < number of training
            images. Default is 0 and no validation set is created.
        gt_downscale_factor: int
            Scale factor specifying the spatial size of square GT maps in relation to the input. For instance,
            `gt_downscale_factor` = 4 means that the spatial size of returned `gt_head_maps` (gtH, gtW) is exactly
            one-fourth that of input `images` (H, W) [see details of train_get_data() and test_get_data() functions for
            more information]. The value MUST BE one of [1, 2, 4, 8, 16, 32].
        """
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max
        self.image_crop_size = image_crop_size
        assert(self.image_crop_size >= self.image_size_min >= 224)
        assert(self.image_size_max > self.image_size_min)
        self.data_path = data_path
        self.name = name
        self.gt_downscale_factor = gt_downscale_factor

        self.density_map_kernel = util._gaussian_kernel(density_map_sigma)
        self._image_size_multiple = image_size_multiple if image_size_multiple > 0 else gt_downscale_factor
        assert(self.image_size_min % self._image_size_multiple == 0)
        assert(self.image_size_max % self._image_size_multiple == 0)
        assert(self.image_crop_size % self._image_size_multiple == 0)
        self.train_iterator = None
        self.data_paths = {
                            'train': {
                                        'images': os.path.join(self.data_path, 'train_data', 'images'),
                                        'gt': os.path.join(self.data_path, 'train_data', 'ground_truth')
                                     },
                            'test': {
                                        'images': os.path.join(self.data_path, 'test_data', 'images'),
                                        'gt': os.path.join(self.data_path, 'test_data', 'ground_truth')
                                    }
                          }

        self.data_files = {
                            'train': [f for f in sorted(os.listdir(self.data_paths['train']['images']))
                                      if os.path.isfile(os.path.join(self.data_paths['train']['images'], f))],
                            'test': [f for f in sorted(os.listdir(self.data_paths['test']['images']))
                                     if os.path.isfile(os.path.join(self.data_paths['test']['images'], f))]
                          }

        self.num_train_images = len(self.data_files['train'])
        self.num_test_images = len(self.data_files['test'])
        assert(valid_set_size < self.num_train_images)
        self.num_val_images = valid_set_size
        assert(self.num_train_images > 0 and self.num_test_images > 0)
        print('In CrowdDataset.__init__(): {} train and {} test images.'.format(self.num_train_images,
                                                                                self.num_test_images))
        if valid_set_size > 0:
            files = self.data_files['train']
            files_selected = random.sample(range(0, len(files)), valid_set_size)
            validation_files = [f for i, f in enumerate(files)
                                if i in files_selected]
            train_files = [f for i, f in enumerate(files)
                           if i not in files_selected]
            self.data_paths['test_valid'] = self.data_paths['train']
            self.data_files['test_valid'] = validation_files
            self.data_files['train'] = train_files
            self.num_train_images = len(self.data_files['train'])
            print('In CrowdDataset.__init__(): {} valid images selected and train set reduces to {}.'
                  .format(len(self.data_files['test_valid']), len(self.data_files['train'])))
            
        print('In CrowdDataset.__init__(): {} dataset initialized.'.format(self.name))

    def train_get_data(self, batch_size=4, return_names = False):
        """
        Returns a batch of randomly cropped images from train set (with flip augmentation).

        Parameters
        ----------
        batch_size: int
            Required batch size.

        Returns
        ----------
        List of [images: ndarray((B, C, H, W)),
                 gt_head_maps: ndarray((B, 1 + 1{background}, gtH, gtW)),
                 gt_box_maps_list:
                        [ndarray((B, self.num_boxes_per_scale + 1{background}, gtH, gtW)),
                         ndarray((B, self.num_boxes_per_scale + 1{background}, gtH // (2 ** 1), gtW // (2 ** 1))),
                         ndarray((B, self.num_boxes_per_scale + 1{background}, gtH // (2 ** 2), gtW // (2 ** 2))), ...,
                         ndarray((B, self.num_boxes_per_scale + 1{background}, gtH // (2 ** (self.num_scales - 1)),
                                                                               gtW // (2 ** (self.num_scales - 1))))]
        where (gtH, gtW) = (H, W) // self.gt_downscale_factor.
        """
        assert(batch_size > 0)
        
        # randomly sample train dataset
        files = self.data_files['train']
        if self.train_iterator is None or (self.train_iterator + batch_size) > self.num_files_rounded:
            self.train_iterator = 0
            self.num_files_rounded = len(files) - (len(files) % batch_size)
            self.file_ids = random.sample(range(0, len(files)), self.num_files_rounded)
        
        file_ids = self.file_ids[self.train_iterator: self.train_iterator + batch_size]
        assert(len(file_ids) == batch_size)
        file_batch = [files[i] for i in file_ids]
        self.train_iterator += batch_size
        
        # initialize train batch
        num_channels = 3
        images = np.empty((batch_size, num_channels, self.image_crop_size, self.image_crop_size), dtype=np.float32)
        gt_crop_size = self.image_crop_size // self.gt_downscale_factor
        gt_density_maps = np.empty((batch_size, 1, gt_crop_size, gt_crop_size), dtype=np.float32)
        flip_flags = np.random.randint(2, size=batch_size)
        full_image_count = []
        # create batch
        for i, (file_name, flip_flag) in enumerate(zip(file_batch, flip_flags)):
            #print(file_name)
            image, gt_head_map = self._read_image_and_gt_map(file_name, self.data_paths['train']['images'],
                                                             self.data_paths['train']['gt'])

            h, w = image.shape[1] // self.gt_downscale_factor, image.shape[2] // self.gt_downscale_factor
            gt_density_map = util._create_heatmap((image.shape[1], image.shape[2]), (h, w),
                                                    gt_head_map, self.density_map_kernel)
            gt_density_map = gt_density_map[np.newaxis, ...]

            if flip_flag == 1:
                image = image[:, :, :: -1]
                gt_density_map = gt_density_map[:, :, :: -1]

            y, x = 0, 0
            # random draw (y, x) and make multiple of self._image_size_multiple
            if image.shape[1] != self.image_crop_size:
                y = (np.random.randint(image.shape[1] - self.image_crop_size) // self._image_size_multiple) \
                    * self._image_size_multiple
            if image.shape[2] != self.image_crop_size:
                x = (np.random.randint(image.shape[2] - self.image_crop_size) // self._image_size_multiple) \
                    * self._image_size_multiple
            images[i, :, :, :] = image[:, y: y + self.image_crop_size, x: x + self.image_crop_size]
            y //= self.gt_downscale_factor
            x //= self.gt_downscale_factor

            gt_density_maps[i, 0, :, :] = gt_density_map[:, y: y + gt_crop_size, x: x + gt_crop_size]
            full_image_count.append(gt_density_map.sum())
        assert(np.all(np.logical_and(0.0 <= images, images <= 255.0)))
        full_image_count = np.array(full_image_count)
        assert(full_image_count.shape == (batch_size,))
        if return_names:
            return images, gt_density_maps, full_image_count, file_batch
        else:
            return images, gt_density_maps, full_image_count

    def test_get_data(self, set_name='test'):
        """
        An iterator to run over images of test/valid set.

        Parameters
        ----------
        set_name: string
            Name of the set ('test' or 'test_valid') for evaluation.

        Returns
        ----------
        An iterator which outputs a tuple of 4 items:
        image_name: string
            file name
        image: ndarray((1, 3, H, W))
        gt_head_map: ndarray((1, 1 + 1{background}, gtH, gtW))
        gt_box_map_list: List of
                        [ndarray((1, self.num_boxes_per_scale + 1{background}, gtH, gtW)),
                         ndarray((1, self.num_boxes_per_scale + 1{background}, gtH // (2 ** 1), gtW // (2 ** 1))),
                         ndarray((1, self.num_boxes_per_scale + 1{background}, gtH // (2 ** 2), gtW // (2 ** 2))), ...,
                         ndarray((1, self.num_boxes_per_scale + 1{background}, gtH // (2 ** (self.num_scales - 1)),
                                                                               gtW // (2 ** (self.num_scales - 1))))]
        where (gtH, gtW) = (H, W) // self.gt_downscale_factor.

        Example Usage
        ----------
        for name, image, gt_head_map, gt_box_maps_list in _.test_get_data()
            # process
        """
        assert(set_name in ['test', 'test_valid'])
        for image_name in self.data_files[set_name]:
            image, gt_head_map = self._read_image_and_gt_map(image_name, self.data_paths[set_name]['images'],
                                                             self.data_paths[set_name]['gt'])

            h, w = image.shape[1] // self.gt_downscale_factor, image.shape[2] // self.gt_downscale_factor
            gt_density_map = util._create_heatmap((image.shape[1], image.shape[2]), (h, w),
                                                    gt_head_map, self.density_map_kernel)
            gt_density_map = gt_density_map[np.newaxis, ...]

            yield image_name, image[np.newaxis, :, :, :], gt_density_map[:,np.newaxis,:,:]

    # ### ### ### Internal functions ### ### ### #

    def _read_image_and_gt_map(self, image_name, image_path, gt_path=None):
        """
        Reads image and corresponding ground truth.

        Parameters
        ----------
        image_name: string
            file name
        image_path: string
            directory path to image file
        gt_path: string
            directory path to corresponding gt file

        Returns
        ----------
        image: ndarray(3, H, W)
        gt_head_map: ndarray(1, gtH, gtW)
        """
        image = cv2.imread(os.path.join(image_path, image_name))
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert(len(image.shape) == 3)
        height_orig, width_orig, _ = image.shape
        height, width, _ = image.shape
        
        # setting minimum size
        if height < self.image_size_min or width < self.image_size_min:
            if height <= width:
                width = int((float(self.image_size_min * width) / height) + 0.5)
                height = self.image_size_min
            else:
                height = int((float(self.image_size_min * height) / width) + 0.5)
                width = self.image_size_min
        
        # setting maximum size
        if height > self.image_size_max or width > self.image_size_max:
            if height >= width:
                width = int((float(self.image_size_max * width) / height) + 0.5)
                height = self.image_size_max
            else:
                height = int((float(self.image_size_max * height) / width) + 0.5)
                width = self.image_size_max
        
        # make sizes multiple
        if height % self._image_size_multiple != 0:
            height = ((height // self._image_size_multiple) + 1) * self._image_size_multiple
        if width % self._image_size_multiple != 0:
            width = ((width // self._image_size_multiple) + 1) * self._image_size_multiple
        
        # resize image
        if height != height_orig or width != width_orig:
            image = cv2.resize(src = image, dsize = (width, height))
        image = image.transpose((2, 0, 1)).astype(np.float32) # (3, H, W)
        assert(np.all(np.logical_and(0.0 <= image, image <= 255.0)))
        assert(np.all(np.isfinite(image)))
        if gt_path is None:
            return image, None
        
        # read GT
        f_name, _ = os.path.splitext(image_name)
        if self.name in ['ucfqnrf', 'ucf50']:
            gt_path = os.path.join(gt_path, f_name + '_ann.mat')
        elif self.name in ['parta', 'partb', 'parta_1', 'parta_5', 'parta_10', 'parta_30']:
            gt_path = os.path.join(gt_path, 'GT_' + f_name + '.mat')
        data_mat = scipy.io.loadmat(gt_path)
        if self.name in ['ucfqnrf', 'ucf50']:
            gt_annotation_points = data_mat['annPoints']
        elif self.name in ['parta', 'partb', 'parta_1', 'parta_5', 'parta_10', 'parta_30']:
            gt_annotation_points = data_mat['image_info'][0, 0]['location'][0, 0]
        gt_annotation_points -= 1  # MATLAB indices
        '''
        annotation_points : ndarray Nx2,
                            annotation_points[:, 0] -> x coordinate
                            annotation_points[:, 1] -> y coordinate
        '''
        # scale GT points

        gt_map_shape = (height, width)
        gt_annotation_points[:, 0] *= (float(gt_map_shape[1]) / width_orig)
        gt_annotation_points[:, 1] *= (float(gt_map_shape[0]) / height_orig)

        # remove invalid indices
        indices = (gt_annotation_points[:, 0] < gt_map_shape[1]) & \
                  (gt_annotation_points[:, 0] >= 0) & \
                  (gt_annotation_points[:, 1] < gt_map_shape[0]) & \
                  (gt_annotation_points[:, 1] >= 0)
        gt_annotation_points = gt_annotation_points[indices, :].astype(int)
        
        gt_annotation_points = np.floor(gt_annotation_points)

        gt_head_map = gt_annotation_points
        return image, gt_head_map



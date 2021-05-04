"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
import numpy as np
from data.base_dataset import BaseDataset, get_transform_npy, get_params
import torchvision.transforms as transforms

# from data.image_folder import make_dataset
# from PIL import Image

def make_dataset(dir, max_dataset_size = float("inf")):
    "Make list of paths to A and B files."
    APaths = sorted(os.listdir(os.path.join(dir, 'A')))
    BPaths = sorted(os.listdir(os.path.join(dir, 'B')))
    
    APaths = APaths[:min(max_dataset_size, len(APaths))]
    BPaths = BPaths[:min(max_dataset_size, len(BPaths))]

    # Sanity
    assert APaths == BPaths, 'We expected to find exact equal AB pairs'

    # Appending base path
    APaths = [os.path.join(dir, *['A', f]) for f in APaths]
    BPaths = [os.path.join(dir, *['B', f]) for f in BPaths]

    # Creating a list of tuples of the paths so the remain aligned  

    AB_paths = [(APath, BPath) for APath, BPath in zip(APaths, BPaths)]

    return AB_paths
    

class NpArrayAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset saved as numpy arrays.
    
    It assumes that the directory '/path/to/data/train' contains folders A and B
    containing the {A,B} image of an image pair respectively. The images in 
    a pair is assumed to have the same file name.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     """Add new dataset-specific options, and rewrite default values for existing options.

    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #     Returns:
    #         the modified parser.
    #     """
    #     parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #     parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
    #     return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = make_dataset(self.dir_AB)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # path = 'temp'    # needs to be a string
        # data_A = None    # needs to be a tensor
        # data_B = None    # needs to be a tensor
        # return {'data_A': data_A, 'data_B': data_B, 'path': path}
        # return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        
        A_Path, B_Path = self.AB_paths[index]
        A = np.load(A_Path)
        B = np.load(B_Path)

        # Reshape into [num_ims, channels, x, y]
        # A = np.moveaxis(A, -1, 0)
        # B = np.moveaxis(np.array([B, B, B]), 0, -1)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape[:2])
        A_transform_list = get_transform_npy(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform_list = get_transform_npy(self.opt, transform_params, grayscale=(self.output_nc == 1), scale = False)

        A_transform = transforms.Compose(A_transform_list)
        B_transform = transforms.Compose(B_transform_list)

        A = A_transform(A)
        B = B_transform(B)
        
        return {'A': A, 'B': B, 'A_paths': A_Path, 'B_paths': B_Path}


    def __len__(self):
        """Return the total number of images."""
        return len(self.AB_paths)

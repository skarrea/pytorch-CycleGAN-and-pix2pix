# Branch mission
With this branch we seek to rewrite the dataloader such that it supports 
arbitrary numpy arrays as input and output from the network.

## What needs to be modified?

There are three things that needs to be modified and that we need to make sure 
are working for the code to work the way we want it to.

 - **Loading**
   - We must modify the dataloader such that it is able to read numpy array files
   for us to be able to use whatever bitdepth we want.
- **Write**
  - We want the output to be written to numpy array files. 
- **Visualization**
  - We should make sure that out modifications are compatible with the 
  visualization supplied in the codebase.

## The flow of the data loader

[train.py](train.py) calls ```create_dataset``` from the [data module]('../data/__init__.py'). This function is used to create a custom dataset (e.g aligned, unaligned, ...) by using the 
specied option in the command line call. For the pix-2-pix network that we're 
using we are using the aligned dataset and thus our dataset gets the class ```AlignedDataset```. This is the dataset class that is passed to the pytorch ```DataLoader```, and it is a [map-style dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets).

The 

# To do

 - [x] Check internal representation of the tensor after the image is loaded.
   - Representation is float 64
 - [x] Check internal representation of the terson when the output image is written.
    - Also float64
 - [x] Create a float16 test dataset to work with.
 - [x] Write a custom aligned dataset class for loading the 16 bit numpy arrays.

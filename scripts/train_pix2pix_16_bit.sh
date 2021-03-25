set -ex
python train.py --dataroot datasets/16bitnumpyarrayDatasetCropSplit --name npyArray16bitCrop --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode np_array_aligned --norm batch --pool_size 0  --preprocess none --display_port 9999

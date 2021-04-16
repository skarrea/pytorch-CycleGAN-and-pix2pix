set -ex
python train.py --dataroot datasets/16bitnumpyarrayDatasetValSet --name npyArray16bitValSetResNetBackend --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode np_array_aligned --norm batch --pool_size 0  --preprocess none --display_port 9999 --rmse --display_ewma_loss --display_ewma_halflife 3 --validate --validation_freq 2000 --batch_size 5

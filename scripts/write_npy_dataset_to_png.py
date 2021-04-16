import sys
sys.path.append('..')
from util.utilBitDepth import full_dynamic_scale, save_image 
from pathlib import Path 
import numpy as np
from PIL import Image
from tqdm import tqdm

datasetPath = Path('../datasets/16bitnumpyarrayDatasetValSetClean') 
outpath = Path( '../datasets/16bitnumpyarrayDatasetValSetCleanpng')

if not outpath.exists():
    outpath.mkdir()

npy_images = list(datasetPath.rglob('*.npy'))

png_images = [outpath / elem.relative_to(datasetPath) for elem in npy_images]
png_images = [Path(str(elem).replace('.npy', '.png')) for elem in png_images]

npy_images.sort()
png_images.sort()


for npy_path, png_path in tqdm(list(zip(npy_images, png_images))):
    npy_im = np.load(npy_path)
    # scale
    png_im = (full_dynamic_scale(npy_im)*255).astype(np.uint8)
    # save
    if not png_path.parent.exists():
        png_path.parent.mkdir(parents=True)
    image_pil = Image.fromarray(png_im)
    image_pil.save(png_path)
    


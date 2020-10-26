import os
import numpy as np
from imageio import imwrite, imread
import matplotlib.pyplot
import random
import urllib.request
import zipfile

def to_uint8(img):
    if img.dtype == np.dtype(np.uint16):
        img = np.clip(img, 0, 65535)
        img = (img / 65535 * 255).astype(np.uint8)
    elif img.dtype == np.dtype(np.float32) or img.dtype == np.dtype(np.float64):
        img = (img * 255).round().astype(np.uint8)
    elif img.dtype != np.dtype(np.uint8):
        raise Exception("Invalid image dtype " + img.dtype)
    return img

def load_image(img_path, img_type='image', resize=None):
    if img_type == 'image':
        if resize==None:
            b = imread(os.path.join(img_path,'nuclei.png'))
            r = imread(os.path.join(img_path,'microtubules.png'))
            g = imread(os.path.join(img_path,'protein.png'))
            image = to_uint8(np.dstack((r,g,b)))/255
        else:
            new_x, new_y = resize
            b = Image.open(os.path.join(img_path,'nuclei.png'))
            b = b.resize((new_x,new_y))
            r = Image.open(os.path.join(img_path,'microtubules.png'))
            r = r.resize((new_x,new_y))
            g = Image.open(os.path.join(img_path,'protein.png'))
            g = g.resize((new_x,new_y))
            image = to_uint8(np.dstack((np.array(r),np.array(g),np.array(b))))/255
    elif img_type == 'mask':
        if resize==None:
            image = imread(os.path.join(img_path,'cell_border_mask.png'))
        else:
            image = Image.open(os.path.join(img_path,'cell_border_mask.png'))
            image = image.resize((new_x,new_y))
            image = np.array(image)
    return image.astype('float32')

def visualize(images, masks, original_image=None, original_mask=None):
    fontsize = 18
    assert len(images) == len(masks)
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, len(images), figsize=(25, 10))
        for i in range(len(images)):
            ax[0,i].imshow(images[i])
            ax[1,i].imshow(masks[i])
    else:
        f, ax = plt.subplots(2, len(images)+1, figsize=(25,10))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
          
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        for i in range(len(images)):  
            ax[0, i+1].imshow(images[i])
            ax[0, i+1].set_title('Transformed image', fontsize=fontsize)
          
            ax[1, i+1].imshow(masks[i])
            ax[1, i+1].set_title('Transformed mask', fontsize=fontsize)

def download_with_url(url_string, download_file_path='hpa_dataset_interactiveML.zip', unzip=True):
    with urllib.request.urlopen(url_string) as response, open(download_file_path, 'wb') as out_file:
        data = response.read() # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(download_file_path))

def load_from_disk(train_dataset='./data/hpa_dataset' , downloaded_file_path='hpa_dataset_interactiveML.zip'):
  if not os.path.exists(train_dataset):
    download_with_url('https://kth.box.com/shared/static/hcnspau5lndyhkkzgv2ygsyq1978qo90.zip')
    os.rename('hpa_dataset_v2', train_dataset)

def save_to_disk(image, save_path):
    imwrite(save_path, image)

def load_sample_pool(hpa_dir, folder='train'): #folder='test'
    imlist = [name for name in os.listdir(os.path.join(hpa_dir, folder)) if not name.startswith('.')]
    sample_pool = []
    for im_name in imlist:  
        img = load_image(os.path.join(hpa_dir, folder, im_name), img_type='image')
        mask = load_image(os.path.join(hpa_dir, folder, im_name), img_type='mask')
        sample_pool.append((img, mask))
    return sample_pool

def get_one_sample(sample_pool): 
    return random.choice(sample_pool)

def add_new_sample(sample, sample_pool):
    image, geojson = sample
    mask = geojson_to_mask(geojson)
    sample_pool.push((image,mask))

    if len(sample_pool) > 100:
        sample_pool_train.pop(0)
    
    return sample_pool
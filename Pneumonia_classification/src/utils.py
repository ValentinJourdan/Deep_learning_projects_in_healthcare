import cv2
import os
import numpy as np
import random
import torch

   
def load_images(folder_path, filenames):
    images = []
    for filename in filenames:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = img[:,:,0] #On garde 1 channel
            img = cv2.resize(img, (128, 128)) #format = (128,128)
            img = np.expand_dims(img, axis=0) #format = (1,128,128)
            images.append(img)
    return images

def load_dataset(dataset_opt):
    path_P = os.path.join(dataset_opt['path'], 'PNEUMONIA')
    path_N = os.path.join(dataset_opt['path'], 'NORMAL')
    image_files_P = [f for f in os.listdir(path_P) if f.endswith('.jpeg')]
    image_files_N = [f for f in os.listdir(path_N) if f.endswith('.jpeg')]

    images_P = load_images(path_P, image_files_P)
    images_N = load_images(path_N, image_files_N)
    images = images_P + images_N
    labels = [1 for i in range(len(images_P))] + [0 for i in range(len(images_N))]

    data = list(zip(images, labels))

    if dataset_opt['shuffle']:
        data = list(zip(images, labels))
        random.shuffle(data)
        images, labels = zip(*data)

    return torch.tensor(images/255.0, dtype=torch.float32), torch.tensor(labels/255.0, dtype=torch.float32)

def build(obj_type, args):
    return obj_type(**args)
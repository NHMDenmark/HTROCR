import xml.etree.ElementTree as ET
import numpy as np
from skimage.io import imread, imsave
from skimage import draw
import os
root_path = '../../data/training_data/Gjentofte_1881-1913_Denmark/images'
root_path2 = '../../data/training_data/Gjentofte_1881-1913_Denmark/lines'

imgs_to_process = []
for file in os.listdir(root_path):
    if file.endswith('.JPG'):
        imgs_to_process.append(file)

imgs_processed = []
for file in os.listdir(root_path2):
    if file.endswith('.jpg'):
        imgs_processed.append(file.split('_line_')[0] + '.JPG')

elements_to_process = list(set(imgs_to_process) - set(imgs_processed))
print(len(imgs_to_process))
print(len(elements_to_process))
print(elements_to_process)
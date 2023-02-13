from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np
from skimage.io import imread
import os
import re


if __name__ == '__main__':
    path = '/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/Gjentofte_1881-1913_Denmark'
    total = 0
    txt_names=[]
    img_names=[]
    for root, dirs, files in os.walk(path):
        if root.endswith('lines-new'):
            for file in files:
                if file.endswith('.jpg'):
                    img_names.append(file)
        elif root.endswith('Gjentofte_1881-1913_Denmark'):
            for file in files:
                if file.endswith('gt_train_updated.txt'):
                    with open(os.path.join(root, file), 'r') as r:
                        lines = r.readlines()
                        for l in lines:
                            txt_names.append(l.split('\t')[0])
    print(len(txt_names))
    print(len(img_names))
    diff = list(set(txt_names).symmetric_difference(set(img_names)))
    # diff = list(set(txt_names) - set(img_names))
    print(diff)
    # with open(os.path.join(path, 'gt_train_updated.txt'), 'r+') as r:
    #     lines = r.readlines()
    #     r.seek(0)
    #     r.truncate()
    #     for line in lines:
    #         if re.search('.+\.jpg\t\"\s\"$', line) or re.search('.+\.jpg\t\"$', line):
    #             continue
    #         r.write(line)
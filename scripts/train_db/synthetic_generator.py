from trdg.generators import GeneratorFromStrings
import numpy as np
import matplotlib.pyplot as plt
import os
from textline_scrapper import (collect_english_texts, collect_danish_texts, collect_location_info)

TRAIN_DB_DIR = '../../data/training_data'

def generate_lines(lines, type, text_type='en'):
    font_dir = 'fonts/{}'.format(text_type)
    machine_text_dir = os.path.join(TRAIN_DB_DIR, '{}-machine-text-{}'.format(text_type, type))
    fonts = [os.path.join(font_dir, p) 
            for p in os.listdir(font_dir) if os.path.splitext(p)[1] == ".ttf"]

    generator = GeneratorFromStrings(
        lines,
        fonts=fonts,
        count=len(lines),
        skewing_angle = 1,
        random_skew = True,
        size = 135
    )
    
    os.makedirs(machine_text_dir, exist_ok=True)
    os.makedirs(os.path.join(machine_text_dir, "image"), exist_ok=True)
    labels = []
    for id, (img, label) in enumerate(generator):
        # Saving pillow images
        labels.append("sample_{}.jpg\t{}".format(id, label))
        img.save(os.path.join(machine_text_dir, 'image', "sample_{}.jpg".format(id)))

    collection = "\n".join(labels)
    with open(os.path.join(machine_text_dir, "gt_train.txt"), 'w') as w:
        w.write(collection)


if __name__ == '__main__':
    generate_lines(collect_english_texts(), 'en', 'dk')
    # generate_lines(collect_danish_texts(), 'dk', 'dk')
    # generate_lines(collect_location_info(), 'coord', 'coord')
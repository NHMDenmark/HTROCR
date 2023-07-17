from trdg.generators import GeneratorFromStrings
import numpy as np
import matplotlib.pyplot as plt
import os
import unicodedata
from fontTools.ttLib import TTFont
from textline_scrapper import (collect_english_texts, collect_danish_texts, collect_location_info)

TRAIN_DB_DIR = '../../data/training_data'

def is_supported(font, text):
    all_supported = True
    for c in set(text):
        for table in font['cmap'].tables:
            if not ord(c) in table.cmap.keys():
                all_supported = False
                print(font)
    return all_supported

def is_supported(font, text):                    
    supported_chars = set()                      
    for table in font['cmap'].tables:            
        supported_chars.update(table.cmap.keys())
    for c in set(text):                          
        if not ord(c) in supported_chars:        
            return False                         
    return True                                  

def get_supported_font(fonts, text):             
    supported = []                               
    text = "öüøäãäõæåßžáňéšč'àâãèêûúūïíìóźçñńść"
    for font_name in fonts:                      
        if is_supported(TTFont(font_name), text):
            supported.append(font_name)          
            print(font_name)
    return supported  

def generate_lines(lines):
    font_dir = 'fonts/{}'.format('all')
    machine_text_dir = os.path.join(TRAIN_DB_DIR, 'fine-tune-machine-text')
    fonts = [os.path.join(font_dir, p) 
            for p in os.listdir(font_dir) if os.path.splitext(p)[1] == ".ttf"]
    os.makedirs(machine_text_dir, exist_ok=True)
    os.makedirs(os.path.join(machine_text_dir, "image"), exist_ok=True)
    labels = []
    for text in lines:
        for font_name in fonts:
            if not is_supported(TTFont(font_name), text):
                continue
            generator = GeneratorFromStrings(
                text,
                fonts=fonts,
                count=1,
                skewing_angle = 1,
                random_skew = True,
                size = 135
            )

            labels.append("sample_{}.jpg\t{}".format(id, generator[1]))
            generator[0].save(os.path.join(machine_text_dir, 'image', "sample_{}.jpg".format(id)))

    collection = "\n".join(labels)
    with open(os.path.join(machine_text_dir, "gt_train.txt"), 'w') as w:
        w.write(collection)


if __name__ == '__main__':
    # generate_lines(collect_english_texts())
    # generate_lines(collect_danish_texts())
    generate_lines(collect_location_info())
    with open("data/machine.txt", 'r') as r:
        lines = r.readlines()
        generate_lines(lines, '')

    # Tests to check which fonts support unique special characters:

    # font_dir = 'fonts/{}'.format('all')
    # fonts = [os.path.join(font_dir, p) 
    #         for p in os.listdir(font_dir) if os.path.splitext(p)[1] == ".ttf"]
    # get_supported_font(fonts,'')
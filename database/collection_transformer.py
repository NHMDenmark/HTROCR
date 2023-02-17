import os
import json
import re
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from line_processor import segment_lines


def format_emunch_textline(line):
    '''
    Remove supplementary annotations that do not match the gt.
    '''
    l = line.strip()
    l = re.sub(r"{\s\.\.\.\s}", "", l)
    l = re.sub(r"{\s\…\s}", "", l)
    l = re.sub(r"\\", "", l)
    l = re.sub(r"\/", "", l)
    l = re.sub(r"‹", "", l)
    l = re.sub(r"›", "", l)
    l = re.sub(r"½", " 1/2", l)
    if re.search('\s*\.\.\.\s*', l) or \
       re.search('\s*\…s*', l) or \
       re.search('{\w+?}', l):
        return l, True
    formatted_lines = re.split(r'\s{3,}', l)
    return formatted_lines, False

def extract_line_images(file, lines_dir_path, root, params):
    '''
    Read full image, segment into lines and save generated line level images
    '''
    print("Segmenting img {}".format(file))
    img_to_read = os.path.join(root, file)
    orig_image = img_as_ubyte(imread(img_to_read))
    line_imgs = segment_lines(orig_image, lines_dir_path, params)
    for index, img in enumerate(line_imgs, 1):
        imsave(os.path.join(lines_dir_path, 
                            "{}_line_{}.jpg".format(img_to_read.replace('.jpg', ''),
                                                    index)), img)

def extract_line_gt(file, output, ql, root):
    counter = 1
    with open(os.path.join(root, file), 'r') as r:
        lines = r.readlines()
        for l in lines:
            formatted_lines, questionable = format_emunch_textline(l)
            if questionable:
                ql += 'Questionable_{}_line_{}.jpg\t{}\n'.format(file, counter, formatted_lines)
                counter += 1
            else:
                for formatted_line in formatted_lines:
                    if formatted_line != '':
                        output += \
                        '{}_line_{}.jpg\t{}\n'.format(file, counter, formatted_line)
                        counter += 1
    return output, ql


def process_emunch_collection(path):
    '''
    Prepare tuning parameters, run line cropping algorithm
    and extract lines
    '''
    with open('config/emunch.json') as f:
        params = json.load(f)

    for root, dirs, files in os.walk(path):
        gt_path = ''
        lines_dir_path = ''
        output = ""
        questionable_list = ""
        if 'images' in dirs and 'labels' in dirs:
            lines_dir_path = os.path.join(root, 'lines')
            gt_path = root
            print("Directory: {}".format(root))
            os.makedirs(lines_dir_path, exist_ok=True)
        for file in files:
            if file.endswith('.jpg'):
                extract_line_images(file, lines_dir_path, root, params)
            elif file.endswith('.txt'):
                output, questionable_list = extract_line_gt(file, output, questionable_list, root)
    
        with open(os.path.join(gt_path, 'gt_train.txt'), 'w') as w:
            w.write(output)
        with open(os.path.join(gt_path, 'questionable.txt'), 'w') as w:
            w.write(questionable_list)



if __name__ == '__main__':
    process_emunch_collection("../../data/training_data/emunch")

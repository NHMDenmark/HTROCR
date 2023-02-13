from collectiontransformers.CollectionTransformer import CollectionTransformer
import json
import re
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from line_processor import segment_lines
import os


class RetroTransformer(CollectionTransformer):
    def __init__(self, path):
        super().__init__(path)
        with open('config/retro.json') as f:
            self.params = json.load(f)

    def extract_line_images(self, file, lines_dir_path, root):
        '''
        Filter out first and last images - due to page background non-text features
        are getting picked up.
        '''
        img_to_read = os.path.join(root, file)
        orig_image = img_as_ubyte(imread(img_to_read))
        line_imgs = segment_lines(orig_image, lines_dir_path, self.params)
        for index, img in enumerate(line_imgs):
            if index == 0 or index == len(line_imgs)-1:
                continue
            loc = os.path.join(lines_dir_path,
                                "{}_line_{}.jpg".format(file.replace('.jpg', ''),
                                                        index))
            imsave(loc, img)

    def format_textline(self, line):
        '''
        Text seems to be clean of questionable parts - no formatting needed.
        '''
        return line.strip(), False
    
    def extract_line_gt(self, file, output, ql, root):
        counter = 1
        with open(os.path.join(root, file), 'r') as r:
            lines = r.readlines()
            if len(lines) == 1 and lines[0] == ' \n':
                return output, ql
            for l in lines:
                # Collections usually contain gt text specifics - separate unknown lines
                formatted_line, questionable = self.format_textline(l)
                if questionable:
                    ql += 'Questionable_{}_line_{}.jpg\t{}\n'.format(file.replace('.txt', ''), counter, formatted_line)
                    counter += 1
                else:
                    if formatted_line != '':
                        output += \
                        '{}_line_{}.jpg\t{}\n'.format(file.replace('.txt', ''), counter, formatted_line)
                        counter += 1
        return output, ql
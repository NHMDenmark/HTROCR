from abc import ABC, abstractmethod
import os
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from htrocr.database.line_processor import segment_lines
import json
import time

class CollectionTransformer(ABC):
    def __init__(self, path):
        super().__init__()
        with open('config/default.json') as f:
            self.params = json.load(f)
        self.path = path


    @abstractmethod
    def format_textline(self, line):
        '''
        Collection specific gt formatting.
        '''
        pass

    def extract_line_images(self, file, lines_dir_path, root):
        img_to_read = os.path.join(root, file)
        orig_image = img_as_ubyte(imread(img_to_read))
        line_imgs = segment_lines(orig_image, lines_dir_path, self.params)
        for index, img in enumerate(line_imgs, 1):
            loc = os.path.join(lines_dir_path,
                                "{}_line_{}.jpg".format(file.replace('.jpg', ''),
                                                        index))
            imsave(loc, img)

    def extract_line_gt(self, file, output, ql, root):
        counter = 1
        with open(os.path.join(root, file), 'r') as r:
            lines = r.readlines()
            for l in lines:
                # Collections usually contain gt text specifics - separate unknown lines
                formatted_lines, questionable = self.format_textline(l)
                if questionable:
                    ql += 'Questionable_{}_line_{}.jpg\t{}\n'.format(file.replace('.txt', ''), counter, formatted_lines)
                    counter += 1
                else:
                    for formatted_line in formatted_lines:
                        if formatted_line != '':
                            output += \
                            '{}_line_{}.jpg\t{}\n'.format(file.replace('.txt', ''), counter, formatted_line)
                            counter += 1
        return output, ql

    def process_collection(self):
        gt_path = ''
        lines_dir_path = ''
        output = ""
        questionable_list = ""
        start = 0
        for root, dirs, files in os.walk(self.path):
            if 'images' in dirs and 'labels' in dirs:
                start = int(round(time.time()))
                print("Processing directory: {}".format(root))
                lines_dir_path = os.path.join(root, 'lines')
                gt_path = root
                # os.makedirs(lines_dir_path, exist_ok=True)
            # if root.endswith('images'):
            #     for file in files:
            #         if file.endswith('.jpg'):
            #             self.extract_line_images(file, lines_dir_path, root)
                lines_dir_path = ''
                print('Finished cropping images. Elapsed time: {} seconds'.format(int(round(time.time())) - start))
            elif root.endswith('labels-updated'):
                for file in files:
                    if file.endswith('.txt'):
                        output, questionable_list = self.extract_line_gt(file, output, questionable_list, root)
                with open(os.path.join(gt_path, 'gt_train_updated.txt'), 'w') as w:
                    w.write(output)
                with open(os.path.join(gt_path, 'questionable.txt'), 'w') as w:
                    w.write(questionable_list)
                gt_path = ''
                output = ''
                questionable_list = ''
        print("Done.")
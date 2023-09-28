import os
from PIL import Image

def crop_labels(path):
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and filename.startswith('sp'):
            im = Image.open(f)
            width, height = im.size
            left = 0
            top = 0
            right = width-570
            bottom = height
            print('Cropping image {}'.format(f))
            im1 = im.crop((left, top, right, bottom))
            im1.save(f)

if __name__ == '__main__':
    # crop_labels("../NHMD_ORIG_100_cropped")
    print(os.path.splitext("../NHMD_ORIG_100_cropped/sp62211050387356490252.att.jpg")[0] + '.txt')
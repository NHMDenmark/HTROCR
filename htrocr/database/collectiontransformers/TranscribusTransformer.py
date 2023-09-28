from htrocr.database.collectiontransformers.CollectionTransformer import CollectionTransformer
# from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from skimage.exposure import is_low_contrast
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
# from skimage import draw
from PIL import Image, ImageDraw
from tqdm import tqdm

ns = {'d': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

class TranscribusTransformer(CollectionTransformer):
    def __init__(self, path):
        super().__init__(path)

    def format_textline(self, line):
        '''
        Text seems to be clean of questionable parts - no formatting needed.
        '''
        return line, False

    def process_line_images(self, file, tr_output, lines_dir_path, root):
        """
        Extracts bounding box that is wrapping the given polygon coordinates
        """
        img_to_read = os.path.join(root, file)
        orig_image = imread(img_to_read)
        img_xml_tree = ET.parse(os.path.join(root, '../page', file.replace('.jpg', '.xml')))
        img_xml_root = img_xml_tree.getroot()
        text_lines = img_xml_root.findall(".//d:TextLine", ns)
        for index, text_line in enumerate(text_lines):
            if text_line.find('d:TextEquiv/d:Unicode', ns) != None:
                transcription = text_line.find('d:TextEquiv/d:Unicode',ns).text
                points_string = text_line.find('d:Coords', ns).attrib['points']
                coords = points_string.split(' ')
                coords = [tuple(map(int, coord.split(','))) for coord in coords if ',' in coord]
                if len(coords) == 0:
                    continue
                points = np.array(coords)
                img = orig_image.copy()
                cropped_image = img[np.min(points[:,1]):np.max(points[:,1]), np.min(points[:,0]):np.max(points[:,0])]
                if np.any(np.array(cropped_image.shape) == 0) or is_low_contrast(cropped_image):
                    continue
                line_image_filename = "{}_line_{}.jpg".format(file.replace('.jpg', ''), index)
                loc = os.path.join(lines_dir_path, line_image_filename)
                imsave(loc, cropped_image, check_contrast=False)
                tr_output += "{}\t{}\n".format(line_image_filename, transcription)
        return tr_output

    def process_line_images_polygons(self, file, tr_output, lines_dir_path, root):
        """
        Extracts polygons based on xml coordinates
        """
        img_to_read = os.path.join(root, file)
        pil_image = Image.open(img_to_read)
        orig_image = np.array(pil_image) # imread(img_to_read)
        # Select average color for bounding box outside polygon
        average_color = np.mean(orig_image, axis=(0, 1))
        if len(orig_image.shape) == 3:
            color = [np.uint8(x) for x in average_color]
        else:
            color = average_color
        img_xml_tree = ET.parse(os.path.join(root, '../page', file.replace('.jpg', '.xml')))
        img_xml_root = img_xml_tree.getroot()
        text_lines = img_xml_root.findall(".//d:TextLine", ns)
        for index, text_line in enumerate(text_lines):
            if text_line.find('d:TextEquiv/d:Unicode', ns) != None:
                transcription = text_line.find('d:TextEquiv/d:Unicode',ns).text
                points_string = text_line.find('d:Coords', ns).attrib['points']
                coords = points_string.split(' ')
                coords = [tuple(map(int, coord.split(','))) for coord in coords]
                points = np.array(coords)
                # Extract polygon
                mask = Image.new("1", pil_image.size,0)
                ImageDraw.Draw(mask).polygon(list(zip(points[:,0], points[:,1])), fill=1)
                mask_array = np.array(mask)
                # Wrap polygon with a bounding box
                masked_image = np.ones_like(orig_image)*color
                if len(orig_image.shape) == 3:
                    masked_image[mask_array, :] = orig_image[mask_array, :]
                else:
                    masked_image[mask_array] = orig_image[mask_array]
                masked_image = masked_image[np.min(points[:,1]):np.max(points[:,1]), np.min(points[:,0]):np.max(points[:,0])]
                # If using skimage:

                # mask = np.zeros(orig_image.shape[:2], dtype=bool)
                # rr, cc = draw.polygon(points[:, 1], points[:, 0], shape=orig_image.shape[:2])
                # mask[rr, cc] = True
                # masked_image = img_as_ubyte(np.ones_like(orig_image)*color)
                # masked_image[mask, :] = orig_image[mask, :]
                # masked_image = masked_image[np.min(rr):np.max(rr), np.min(cc):np.max(cc)]
                line_image_filename = "{}_line_{}.jpg".format(file.replace('.jpg', ''), index)
                loc = os.path.join(lines_dir_path, line_image_filename)
                if len(orig_image.shape) == 3:
                    res = Image.fromarray(masked_image)
                    res.save(loc)
                else:
                    res = Image.fromarray(masked_image).convert('L')
                    res.save(loc)
                # imsave(loc, masked_image, check_contrast=False)
                tr_output += "{}\t{}\n".format(line_image_filename, transcription)
        return tr_output

    
    def process_collection(self):
        """
        Process Transkribus zip exports with '.jpg' format images. 
        It is assumed that all images are placed within `images` dir.
        """
        gt_path = ''
        lines_dir_path = ''
        output = ""
        start = 0
        for root, dirs, files in os.walk(self.path):
            # Making an assumption of file structure
            if 'images' in dirs and 'page' in dirs:
                start = int(round(time.time()))
                print("Processing directory: {}".format(root))
                lines_dir_path = os.path.join(root, 'lines')
                gt_path = root
                if os.path.exists(lines_dir_path):
                    continue
                os.makedirs(lines_dir_path, exist_ok=True)
            if root.endswith('images'):
                for i in tqdm(range(len(files)), desc="pages"):
                    file = files[i]
                    # Process only '.jpg' files
                    if file.endswith('.jpg'):
                        # alternatively a slower, but more precise segmentation without upper, lower line noise: 
                        # output = self.process_line_images_polygons(file, output, lines_dir_path, root)
                        output = self.process_line_images(file, output, lines_dir_path, root)
                with open(os.path.join(gt_path, 'gt_train.txt'), 'w') as w:
                    w.write(output)
                print('Finished cropping images. Elapsed time: {} seconds'.format(int(round(time.time())) - start))
                gt_lines = output.splitlines()
                no_of_lines_gt = len(gt_lines)
                line_images = [img for img in os.listdir(lines_dir_path) if img.endswith('.jpg')]
                no_of_images = len(line_images)
                print("Number of GT lines: ", no_of_lines_gt)
                print("Number of line images: ", no_of_images)
                if no_of_images != no_of_images:
                    difference = list(set(no_of_images).symmetric_difference(set(no_of_lines_gt)))
                    print("Differences:" )
                    print(difference)
                output = ""
        print("Done.")

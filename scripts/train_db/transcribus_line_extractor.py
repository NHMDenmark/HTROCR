import xml.etree.ElementTree as ET
import numpy as np
from skimage.io import imread, imsave
from skimage import draw
import os
root_path = '/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/Gjentofte_1881-1913_Denmark/images'
root_path2 = '/Users/linas/Studies/UCPH-DIKU/thesis/code/data/training_data/Gjentofte_1881-1913_Denmark/lines'

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

# tree = ET.parse("Ahvenanmaan_1869-1870_10.xml")
# root = tree.getroot()
# ns = {'d': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
# text_lines = root.findall(".//d:TextLine", ns)

# # iterate over the TextLine elements and access their data
# # for text_line in text_lines:
# print(text_lines[3].find('d:TextEquiv/d:Unicode',ns).text)
# img = imread('Ahvenanmaan_1869-1870_10.jpg')
# points_string = text_lines[3].find('d:Coords',ns).attrib['points']
# coords = points_string.split(' ')

# # convert each string into a tuple of integers
# coords = [tuple(map(int, coord.split(','))) for coord in coords]

# # convert the list of tuples into a numpy array
# points = np.array(coords)

# # Plot the image and the bounding box
# rows, cols = draw.polygon(points[:, 1], points[:, 0], (img.shape[0], img.shape[1]))

# # Crop the image to the polygon
# cropped_image = img[np.min(rows):np.max(rows), np.min(cols):np.max(cols)]

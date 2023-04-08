from LineSegmenter import LineSegmenter
import numpy as np
from scipy.ndimage import grey_dilation
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from util import get_point_neighbors
from skimage.color import rgb2gray

# Height Based Line Segmenter
class HBLineSegmenter(LineSegmenter):
    def __init__(self, path="./config/default.json"):
        super().__init__(path)
    
    def __get_line_bbox(self, Si):
        polygon_set = []
        sum_of_dists = 0
        min_dist = np.inf
        for _, v in Si.items():
            sum_of_dists += v[1]
            min_dist = min(min_dist, v[1])
        mean_dist = sum_of_dists/len(Si)
        for p, (angle_rad, _) in Si.items():
            polygon_set.append(p)
            angle_rad += np.pi/2
            end_point = (int(p[0] - np.sin(angle_rad) * int(mean_dist)), int(p[1] + np.cos(angle_rad) * int(mean_dist)))
            polygon_set.append(end_point)
        sorted_points = self.__convex_hull_polygon(polygon_set)
        minx = min(sorted_points[:,1])-10
        maxx = max(sorted_points[:,1])+20
        miny = min(sorted_points[:,0])-20
        maxy = max(sorted_points[:,0])+20
        return minx, miny, maxx, maxy, int(min_dist)

    def __generate_bbox(bbox_img, points):
        mask = Image.new("1", bbox_img.size, 0)
        ImageDraw.Draw(mask).polygon(list(zip(points[:,0], points[:,1])), fill=1)
        average_color = np.mean(bbox_img, axis=(0, 1))
        color = np.uint8(average_color)
        mask_array = np.array(mask)
        masked_image = np.ones_like(bbox_img)*color
        masked_image[mask_array] = bbox_img[mask_array]
        return masked_image

    def segment_lines(self, img_path):
        scale = 0.33
        orig = Image.open(img_path)
        width, height = orig.size
        new_size = (int(width * scale), int(height * scale))
        resized_img = orig.resize(new_size)
        img = rgb2gray(np.array(resized_img))
        clusters, N = self.bbuilder.run(img)
        segmentations = []
        for i in range(1, len(clusters)):
            Si = clusters[i]
            minx, miny, maxx, maxy, min_region = self.__get_line_bbox(Si)
            label = img[miny:maxy+20, minx:maxx]/255
            polygon_set = []
            mind = np.inf
            for p, (angle_rad, pixels) in Si.items():
                mind = min(mind, pixels)
            for p, (angle_rad, pixels) in Si.items():
                start_point = (p[1]-minx,p[0]-miny)
                descender_point = (int(start_point[0] + np.cos(angle_rad-np.pi/2) * int(20)), int(start_point[1] - np.sin(angle_rad-np.pi/2) * int(20)))
                polygon_set.append(descender_point)
                ascender_point = (int(start_point[0] + np.cos(angle_rad+np.pi/2) * int(mind)), int(start_point[1] - np.sin(angle_rad+np.pi/2) * int(mind)))
                polygon_set.append(ascender_point)
            hull = ConvexHull(polygon_set)
            sorted_points = [tuple(polygon_set[i]) for i in hull.vertices]
            bbox = self.__generate_bbox(label, sorted_points)
            segmentations.append(bbox)
        return segmentations
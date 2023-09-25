from htrocr.line_segmentation.LineSegmenter import LineSegmenter
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from skimage.color import rgb2gray

# Height Based Line Segmenter
class HBLineSegmenter(LineSegmenter):
    def __init__(self, path):
        super().__init__(path)

    def __convex_hull_polygon(self, polygon_set):
        hull = ConvexHull(polygon_set)
        return np.array([polygon_set[i] for i in hull.vertices])

    def __get_line_bbox(self, Si, img):
        polygon_set = []
        sum_of_dists = 0
        min_dist = np.inf
        for _, v in Si.items():
            sum_of_dists += v[1]
            min_dist = min(min_dist, v[1])
        polygon_set = []
        for p, (angle_rad, _) in Si.items():
            start_point = (p[0], p[1])
            # extend point in direction orthogonally bellow the baseline 
            descender_point = (int(start_point[0] - np.sin(angle_rad-np.pi/2) * int(20)), int(start_point[1] + np.cos(angle_rad-np.pi/2) * int(20)))
            polygon_set.append(descender_point)
            # extend point in direction orthogonally above the baseline
            ascender_point = (int(start_point[0] - np.sin(angle_rad+np.pi/2) * int(min_dist)), int(start_point[1] + np.cos(angle_rad+np.pi/2) * int(min_dist)))
            polygon_set.append(ascender_point)
        sorted_points = self.__convex_hull_polygon(polygon_set)
        # Get the bbox but make sure the crop does not go outside image boundaries
        minx = max(min(sorted_points[:,1]), 1)
        maxx = min(max(sorted_points[:,1]), img.shape[1]-1)
        miny = max(min(sorted_points[:,0]), 1)
        maxy = min(max(sorted_points[:,0]), img.shape[0]-1)
        return minx, miny, maxx, maxy, sorted_points

    def segment_lines(self, img_path):
        scale = self.config["downsize_scale"]
        orig = Image.open(img_path)
        width, height = orig.size
        if self.config["crop_ucph_border"]:
            orig = orig.crop((0, 0, width-self.config["crop_ucph_border_size"], height))
        new_size = (int(width * scale), int(height * scale))
        resized_img = orig.resize(new_size)
        img = rgb2gray(np.array(resized_img))
        print(img.size)
        clusters, N = self.bbuilder.run(img)
        segmentations = []
        region_coords = []
        for i in range(len(clusters)):
            Si = clusters[i]
            minx, miny, maxx, maxy, sorted_points = self.__get_line_bbox(Si, img)
            label = img[miny:maxy,minx:maxx]
            segmentations.append(label)
            region_coords.append([minx, maxx, miny, maxy])
        return segmentations, sorted_points, clusters, region_coords, scale

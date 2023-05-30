from line_segmentation.LineSegmenter import LineSegmenter
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage.color import rgb2gray

# Height Based Line Segmenter
class HBLineSegmenter(LineSegmenter):
    def __init__(self, path="./line_segmentation/config/default.json"):
        super().__init__(path)

    def __convex_hull_polygon(self, polygon_set):
        hull = ConvexHull(polygon_set)
        return np.array([polygon_set[i] for i in hull.vertices])

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

    def __generate_bbox(self, bbox_img, points):
        mask = Image.new("1", Image.fromarray(bbox_img).size, 0)
        ImageDraw.Draw(mask).polygon(list(zip(points[:,0], points[:,1])), fill=1)
        average_color = np.mean(bbox_img, axis=(0, 1))
        color = np.uint8(average_color)
        mask_array = np.array(mask)
        masked_image = np.ones_like(bbox_img)*average_color
        masked_image[mask_array] = bbox_img[mask_array]
        masked_image = masked_image[int(np.min(points[:,1])):int(np.max(points[:,1])), int(np.min(points[:,0])):int(np.max(points[:,0]))]
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
        region_coords = []
        for i in range(1, len(clusters)):
            Si = clusters[i]
            minx, miny, maxx, maxy, min_region = self.__get_line_bbox(Si)
            maxx = min(img.shape[1]-1, maxx)
            maxy = min(img.shape[0]-1, maxy)
            minx = max(1, minx)
            miny = max(1, miny)
            label = img[miny:maxy+20, minx:maxx]
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
            sorted_points = np.array([tuple(polygon_set[i]) for i in hull.vertices])
            bbox = self.__generate_bbox(label, sorted_points)
            segmentations.append(bbox)
            max_x = max(sorted_points[:,1])
            min_x = min(sorted_points[:,1])
            max_y = max(sorted_points[:,0])
            min_y = min(sorted_points[:,0])
            region_coords.append([min_x, max_x, min_y, max_y])
        return segmentations, sorted_points, clusters, region_coords, scale

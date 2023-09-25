from htrocr.line_segmentation.LineSegmenter import LineSegmenter
import numpy as np
from scipy.ndimage import grey_dilation
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from htrocr.line_segmentation.util import get_point_neighbors
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import math
from math import acos, degrees

class PreciseLineSegmenter(LineSegmenter):
    def __init__(self, path):
        super().__init__(path)

    def __convex_hull_polygon(self, polygon_set):
        hull = ConvexHull(polygon_set)
        return np.array([polygon_set[i] for i in hull.vertices])

    def __get_line_height(self, Si, N, bbox, min_interpdist, minx, miny):
        """
        Approximately estimates handwritten line height.
        """
        heights = []
        for p, v in Si.items():
            y,x = p
            nvs = get_point_neighbors(N, p)
            y -= miny
            x -= minx
            column=[[x,y]]
            # Form a column using neighbours from the same line
            for nv in nvs:
                if tuple(nv) in list(Si.keys()):
                    column.append([nv[1]-minx, nv[0]-miny])
            theta = v[0] + np.pi/2
            end_point = (int(x + np.cos(theta) * int(min_interpdist-1))+1, int(y - np.sin(theta) * int(min_interpdist-1))+1)
            dx, dy = np.cos(theta), -np.sin(theta)
            height = 0
            stop_criteria = 0
            # Go upwards from the baseline until the end goal is reached
            while int(column[0][0]) != end_point[0] and int(column[0][1]) != end_point[1]:
                # Additionally check if any neighbour has reached the border
                for p in column:
                    if int(p[0]) >= bbox.shape[1] or int(p[1]) >= bbox.shape[0] or int(p[0]) <= 0 or int(p[1]) <= 0:
                        stop_criteria = -1
                        break
                if stop_criteria == -1:
                    break
                exists = any([bbox[int(p[1]), int(p[0])] for p in column])
                if exists:
                    height += 1
                    stop_criteria = 0
                else:
                    stop_criteria += 1
                if stop_criteria >= 30:
                    break
                column = [[p[0]+dx, p[1]+dy] for p in column]
            if height != 0:
                heights.append(height)
        if len(heights) > 0:
            return int(np.array(heights).mean())
        else:
            return 0

    def __add_core_region(self, img, baseline, minx, miny, height):
        """
        Adds core region bounding box on top of text line features to ensure
        that all elements across the line width are included in segmentation
        """
        polygon_set = []
        for p, (angle_rad, _) in baseline.items():
            p0 = p[0]
            p1 = p[1]
            p0 -= miny
            p1 -= minx
            pixels = self.config['min_textline_height']
            start_point = (p1,p0)
            polygon_set.append(start_point)
            angle_rad += np.pi/2
            end_point = (int(p1 + np.cos(angle_rad) * int(max(height, pixels))), int(p0 - np.sin(angle_rad) * int(max(height, pixels))))
            
            polygon_set.append(end_point)
        hull = ConvexHull(polygon_set)
        sorted_points = [tuple(polygon_set[i]) for i in hull.vertices]
        pilImage = Image.fromarray(img)
        
        draw = ImageDraw.Draw(pilImage)
        
        draw.polygon(sorted_points, fill=True)
        return np.array(pilImage)

   

    def __separate_lines(self, baseline, minx, miny, bbox, height):
        """
        Main segmentation steps 
        """
        dilated = self.__add_core_region(bbox, baseline, minx, miny, height)
        dilated = grey_dilation(dilated, footprint=np.ones((3, 3)))
        markers, _ = ndimage.label(dilated)
        return dilated, markers

    def __get_line_bbox(self, Si, img):
        """
        Generates oversized bounding box for precise noise elimination
        """
        polygon_set = []
        sum_of_dists = 0
        min_dist = np.inf
        for _, v in Si.items():
            sum_of_dists += v[1]
            min_dist = min(min_dist, v[1])
        for p, (angle_rad, _) in Si.items():
            start_point = (p[0], p[1])
            # extend point in direction orthogonally bellow the baseline 
            descender_point = (int(start_point[0] - np.sin(angle_rad-np.pi/2) * int(self.config["descender_point_adjuster"])), int(start_point[1] + np.cos(angle_rad-np.pi/2) * int(self.config["descender_point_adjuster"])))
            polygon_set.append(descender_point)
            # extend point in direction orthogonally above the baseline
            ascender_point = (int(start_point[0] - np.sin(angle_rad+np.pi/2) * int(min_dist)), int(start_point[1] + np.cos(angle_rad+np.pi/2) * int(min_dist)))
            polygon_set.append(ascender_point)
        sorted_points = self.__convex_hull_polygon(polygon_set)
        minx = max(min(sorted_points[:,1])-5, 1)
        maxx = min(max(sorted_points[:,1])+5, img.shape[1]-1)
        miny = max(min(sorted_points[:,0])-5, 1)
        maxy = min(max(sorted_points[:,0])+5, img.shape[0]-1)
        
        
        return minx, miny, maxx, maxy, int(min_dist)

    def __merge_line_segments(self, img, baseline, segments):
        """ 
        Merges computed segments that are closest to baseline
        """
        segments_to_merge = []
        region_props = regionprops(segments)
        for p in baseline:
            min_dist = np.inf
            conv_hull = None
            for region in region_props:
                minr, minc, maxr, maxc = region.bbox
                # find closest x and y values to the corresponding p coords.
                closest_x = max(minc, min(p[1], maxc))
                closest_y = max(minr, min(p[0], maxr))
                closest_point = [closest_x, closest_y]
                distance = np.sqrt((p[1] - closest_point[0]) ** 2 + (p[0] - closest_point[1]) ** 2)
                if distance < min_dist:
                    min_dist = distance
                    conv_hull = [region.bbox, region.image_filled]
            segments_to_merge.append(conv_hull)

        binary_image = np.zeros_like(img, dtype=bool)
        polygon_points = np.array([], dtype=np.int64).reshape(0,2)
        for bbox, conv in segments_to_merge:
            minr, minc, maxr, maxc = bbox
            binary_image[minr:maxr, minc:maxc] += conv
            polygon_points = np.vstack([polygon_points, [[minr, minc], [minr, maxc], [maxr, maxc], [maxr,minc]]])
        sorted_points = self.__convex_hull_polygon(polygon_points)
        return binary_image, sorted_points

    def __remove_outliers(self, segmented_img, dilated, sorted_points):
        """
        Removes non line segments (noise)
        """
        other_classes = dilated.copy()
        other_classes[segmented_img] = False
        
        mask = np.zeros_like(segmented_img, dtype=bool)
        mask_img = Image.fromarray(mask)
        
        draw = ImageDraw.Draw(mask_img)
        points= [(s[1], s[0]) for s in sorted_points]
        draw.polygon(points, outline=True, width=2)
        mask_array = np.array(mask_img)
        
        filled_image = binary_fill_holes(mask_array)
        filled_image[filled_image == other_classes] = False
        
        return filled_image

    def __trace_contour(self, filtered_shape):
        """
        Finds the contour on updated shape
        """
        filtered_shape = np.pad(filtered_shape, (self.config["max_contour"],), mode='constant')
        contours = find_contours(filtered_shape)

        if len(contours) > 1:
            sorted_contours = np.array(sorted(contours, key=len, reverse=True), dtype=object)
            bounding_contour = sorted_contours[0]
        else:
            bounding_contour = np.array(contours)[0]
        return bounding_contour

    def __generate_bbox(self, bbox_img, points):
        """
        Constructs a final bounding box
        """
        mask = Image.new("1", Image.fromarray(bbox_img).size, 0)
        ImageDraw.Draw(mask).polygon(list(zip(points[:,1], points[:,0])), fill=1)
        average_color = np.mean(bbox_img, axis=(0, 1))
        mask_array = np.array(mask)
        masked_image = np.ones_like(bbox_img)*average_color        
        masked_image[mask_array] = bbox_img[mask_array]
        y1, y2, x1, x2 = (int(np.min(points[:,0])), int(np.max(points[:,0])), int(np.min(points[:,1])), int(np.max(points[:,1])))
        
        masked_image = masked_image[y1:y2, x1:x2]
        return masked_image, y1, y2, x1, x2

    def rotate_segment(self, segment, boundaryBox):
        xs, xl, xsy, xly = 0, 0, 0, 0
        for coord in segment.keys():                
            if xs > coord[1] or xs == 0:
                xs = coord[1]
                xsy = coord[0]                    
            if xl < coord[1] or xl == 0:
                xl = coord[1]
                xly = coord[0]
                 
        xdis = xl - xs
        ydis = xsy - xly
        aside = ydis
        if aside < 0:
            aside = aside * (-1)
                
        hypo = math.sqrt((aside * aside + xdis * xdis))
            
        rotateAngle = math.degrees(math.acos((xdis * xdis + hypo * hypo - aside * aside)/(2.0 * xdis * hypo)))            
        if ydis > 0:
            rotateAngle = rotateAngle * (-1)
        
        boundaryBox = ndimage.rotate(boundaryBox, rotateAngle, reshape = True, mode=self.config["rotate_angle_mode"], cval=self.config["angle_fill_greyscale"])
        return boundaryBox

    def segment_lines(self, img_path):
        """
        Main driver. Returns segmentation images, contour coordinates (for xmls),
        wrapping bounding box coordinates (region_coords), the image scale value.
        """
        scale = self.config["downsize_scale"]
        orig = Image.open(img_path)
        width, height = orig.size

        if self.config["crop_ucph_border"]:
            orig = orig.crop((0, 0, width - self.config["crop_ucph_border_size"], height))
            
        nwidth, nheight = orig.size
        
        new_size = (int(nwidth * scale), int(nheight * scale))
        resized_img = orig.resize(new_size)
    
        img = rgb2gray(np.array(resized_img))
        
        clusters, N = self.bbuilder.run(img)
        segmentations = []
        polygons = []
        region_coords = []
        sortedClusters = []
        sortMarkers = []
        for i in range(len(clusters)):
            Si = clusters[i]
            minx, miny, maxx, maxy, min_region = self.__get_line_bbox(Si, img)
            label = img[miny:maxy, minx:maxx]
            
            sortMarkYX = 0, 0
            
            for coSets in Si.keys():
                if sortMarkYX[0] > coSets[0] or sortMarkYX[0] == 0:
                    sortMarkYX = coSets[0], coSets[1]
            sortMarkers.append(sortMarkYX)

            sortMarkers.sort()
            impIndex = sortMarkers.index(sortMarkYX)
            
            thresh = threshold_otsu(label)
            thresholded = label < thresh
            height = self.__get_line_height(Si, N, thresholded, min_region, minx, miny)
            
            new_baseline = np.array(list(Si.keys()))
            new_baseline[:,1] -= minx 
            new_baseline[:,0] -= miny
            dilated, segments = self.__separate_lines(Si, minx, miny, thresholded, height)
            binary_image, polygon_points = self.__merge_line_segments(dilated, new_baseline, segments)
            filtered_shape = self.__remove_outliers(binary_image, dilated, polygon_points)
            contour = self.__trace_contour(filtered_shape) - self.config["contour_adjuster"]
            # contour[:, 1] -= self.config["contour_adjuster"]
            bbox, y1, y2, x1, x2 = self.__generate_bbox(label, contour)
            x1 = (x1+minx)/scale 
            x2 = (x2+minx)/scale 
            y1 = (y1+miny)/scale 
            y2 = (y2+miny)/scale
            
            if self.config["use_rotation_angle"]:
                bbox = self.rotate_segment(Si, bbox)

            if self.config["use_border_padding"]:    
                if self.config["border_padding_mode"] == "constant":
                    bbox = np.pad(bbox, (self.config["generate_border_padding_size"], ), mode = "constant", constant_values=self.config["greyscale_for_border_padding"])
                else:
                    bbox = np.pad(bbox, (self.config["generate_border_padding_size"], ), mode = self.config["border_padding_mode"])

            segmentations.insert(impIndex, bbox)
            # segmentations.append(bbox)
             
            polygons.insert(impIndex, [[x1,y1], [x1,y2], [x2,y2], [x2,y1]])
            # polygons.append([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])
            max_x = (max(contour[:,1]) + minx)/scale
            min_x = (min(contour[:,1]) + minx)/scale
            max_y = (max(contour[:,0]) + miny)/scale
            min_y = (min(contour[:,0]) + miny)/scale
            region_coords.insert(impIndex, [min_x, max_x, min_y, max_y])
            # region_coords.append([min_x, max_x, min_y, max_y])
            sortedClusters.insert(impIndex, Si)
            
        return segmentations, polygons, sortedClusters, region_coords, scale
        


from line_segmentation.LineSegmenter import LineSegmenter
import numpy as np
from scipy.ndimage import grey_dilation
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from line_segmentation.util import get_point_neighbors
from skimage.color import rgb2gray

class PreciseLineSegmenter(LineSegmenter):
    def __init__(self, path="./line_segmentation/config/default.json"):
        super().__init__(path)

    def __convex_hull_polygon(self, polygon_set):
        hull = ConvexHull(polygon_set)
        return np.array([polygon_set[i] for i in hull.vertices])

    def __get_line_height(self, Si, N, bbox, min_interpdist, minx, miny):
        heights = []
        for p, v in Si.items():
            y,x = p
            nvs = get_point_neighbors(N, p)
            y -= miny
            x -= minx
            column=[[x,y]]
            for nv in nvs:
                if tuple(nv) in list(Si.keys()):
                    column.append([nv[1]-minx, nv[0]-miny])
            theta = v[0] + np.pi/2
            end_point = (int(x + np.cos(theta) * int(min_interpdist-1)), int(y - np.sin(theta) * int(min_interpdist-1)))
            dx, dy = np.cos(theta), -np.sin(theta)
            height = 0
            stop_criteria = 0
            while int(column[0][0]) != end_point[0] and int(column[0][1]) != end_point[1]:
                exists = any([bbox[int(p[1]), int(p[0])] for p in column])
                if exists:
                    height += 1
                    stop_criteria = 0
                else:
                    stop_criteria += 1
                if stop_criteria >= 30:
                    break
                column = [[p[0]+dx, p[1]+dy] for p in column]
                # x += dx
                # y += dy
            if height != 0:
                heights.append(height)
        if len(heights) > 0:
            return int(np.array(heights).mean())
        else:
            return 0

    def __add_core_region(self, img, baseline, minx, miny, height):
        polygon_set = []
        for p, (angle_rad, _) in baseline.items():
            # Min pixels to separate line segments
            p0 = p[0]
            p1 = p[1]
            p0 -= miny
            p1 -= minx
            pixels = 10
            start_point = (p1,p0)
            polygon_set.append(start_point)
            angle_rad += np.pi/2
            end_point = (int(p1 + np.cos(angle_rad) * int(max(height,pixels))), int(p0 - np.sin(angle_rad) * int(max(height, pixels))))
            polygon_set.append(end_point)
        hull = ConvexHull(polygon_set)
        sorted_points = [tuple(polygon_set[i]) for i in hull.vertices]
        pilImage = Image.fromarray(img)
        draw = ImageDraw.Draw(pilImage)
        draw.polygon(sorted_points, fill=True)
        return np.array(pilImage)

    def __separate_lines(self, baseline, minx, miny, bbox, height):
        dilated = self.__add_core_region(bbox, baseline, minx, miny, height)
        dilated = grey_dilation(dilated, footprint=np.ones((3, 3)))
        dist_transform = ndimage.distance_transform_edt(dilated)
        markers, _ = ndimage.label(dilated)
        return dilated, watershed(-dist_transform, markers, mask=dilated)

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

    def __merge_line_segments(self, img, baseline, watershed_segments):
        # Merge computed segments that are closest to baseline
        segments_to_merge = []
        for p in baseline:
            min_dist = np.inf
            conv_hull = []
            for region in regionprops(watershed_segments):
                minr, minc, maxr, maxc = region.bbox
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
        # Remove non line segments
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
        # Find the contour on updated shape
        filtered_shape = np.pad(filtered_shape, (5,), mode='constant')
        contours = find_contours(filtered_shape)

        if len(contours) > 1:
            sorted_contours = np.array(sorted(contours, key=len, reverse=True), dtype=object)
            bounding_contour = sorted_contours[0]
        else:
            bounding_contour = np.array(contours)[0]
        return bounding_contour

    def __generate_bbox(self, bbox_img, points):
        mask = Image.new("1", Image.fromarray(bbox_img).size, 0)
        ImageDraw.Draw(mask).polygon(list(zip(points[:,1], points[:,0])), fill=1)
        average_color = np.mean(bbox_img, axis=(0, 1))
        mask_array = np.array(mask)
        masked_image = np.ones_like(bbox_img)*average_color
        masked_image[mask_array] = bbox_img[mask_array]
        masked_image = masked_image[int(np.min(points[:,0])):int(np.max(points[:,0])), int(np.min(points[:,1])):int(np.max(points[:,1]))]
        return masked_image

    def segment_lines(self, img_path):
        scale = 0.25
        orig = Image.open(img_path)
        width, height = orig.size
        new_size = (int(width * scale), int(height * scale))
        resized_img = orig.resize(new_size)
        img = rgb2gray(np.array(resized_img))
        clusters, N = self.bbuilder.run(img)
        segmentations = []
        polygons = []
        region_coords = []
        for i in range(1, len(clusters)):
            Si = clusters[i]
            minx, miny, maxx, maxy, min_region = self.__get_line_bbox(Si)
            label = img[miny:maxy, minx:maxx]
            thresh = threshold_otsu(label)
            thresholded = label < thresh
            height = self.__get_line_height(Si, N, thresholded, min_region, minx, miny)
            new_baseline = np.array(list(Si.keys()))
            new_baseline[:,1] -= minx 
            new_baseline[:,0] -= miny
            dilated, watershed_segments = self.__separate_lines(Si, minx, miny, thresholded, height)
            binary_image, polygon_points = self.__merge_line_segments(dilated, new_baseline, watershed_segments)
            filtered_shape = self.__remove_outliers(binary_image, dilated, polygon_points)
            contour = self.__trace_contour(filtered_shape) - 5
            bbox = self.__generate_bbox(label, contour)
            segmentations.append(bbox)
            polygons.append(contour)
            max_x = max(contour[:,1])
            min_x = min(contour[:,1])
            max_y = max(contour[:,0])
            min_y = min(contour[:,0])
            region_coords.append([min_x, max_x, min_y, max_y])
        return segmentations, polygons, clusters, region_coords, scale

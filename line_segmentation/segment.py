import numpy as np
from bbuilder import build_baselines
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


def convex_hull_polygon(polygon_set):
    hull = ConvexHull(polygon_set)
    return np.array([polygon_set[i] for i in hull.vertices])

def get_line_height(Si, bbox, min_interpdist, minx, miny):
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


def separate_lines(label, min_interpdist, height):
    # fsize = max(int(peak_region/3), 9)
    fsize = max(min(height, min_interpdist), 8)
    dilated = grey_dilation(label, footprint=np.ones((fsize, fsize)))
    dist_transform = ndimage.distance_transform_edt(dilated)
    # print(peak_region)
    peak_area_size = int(label.shape[0]/2.5) #min(max(min_interpdist, 20), 61)
    peak_idx = peak_local_max(dist_transform, footprint=np.ones((peak_area_size, peak_area_size), dtype=int), labels=dilated)
    local_maxi = np.zeros_like(dist_transform, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True
    markers, _ = ndimage.label(local_maxi)
    return dilated, dist_transform, watershed(-dist_transform, markers, mask=dilated)

def get_line_bbox(Si):
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
    sorted_points = convex_hull_polygon(polygon_set)
    minx = min(sorted_points[:,1])-10
    maxx = max(sorted_points[:,1])+20
    miny = min(sorted_points[:,0])-20
    maxy = max(sorted_points[:,0])+20
    return minx, miny, maxx, maxy, int(min_dist)

def merge_line_segments(img, baseline, watershed_segments):
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
                conv_hull = [region.bbox, region.image_convex]
        segments_to_merge.append(conv_hull)

    binary_image = np.zeros_like(img, dtype=bool)
    polygon_points = np.array([], dtype=np.int64).reshape(0,2)
    for bbox, conv in segments_to_merge:
        minr, minc, maxr, maxc = bbox
        binary_image[minr:maxr, minc:maxc] += conv
        polygon_points = np.vstack([polygon_points, [[minr, minc], [minr, maxc], [maxr, maxc], [maxr,minc]]])
    sorted_points = convex_hull_polygon(polygon_points)
    return binary_image, sorted_points

def remove_outliers(segmented_img, dilated, sorted_points):
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

def trace_contour(filtered_shape):
    # Find the contour on updated shape
    filtered_shape = np.pad(filtered_shape, (5,), mode='constant')
    contours = find_contours(filtered_shape)

    if len(contours) > 1:
        sorted_contours = np.array(sorted(contours, key=len, reverse=True))
        bounding_contour = sorted_contours[0]
    else:
        bounding_contour = np.array(contours)[0]
    return bounding_contour


def segment_lines(img):
    scale = 0.33
    orig = Image.new(img)
    width, height = orig.size
    new_size = (int(width * scale), int(height * scale))
    resized_img = orig.resize(new_size)
    img = np.array(resized_img)
    clusters, N = build_baselines(img)
    segmentations = []
    for i in range(1, len(clusters)):
        Si = clusters[i]
        minx, miny, maxx, maxy, min_region = get_line_bbox(Si)
        label = img[miny:maxy, minx:maxx]/255
        label = rgb2gray(label)
        thresh = threshold_otsu(label)
        thresholded = label < thresh
        height = get_line_height(Si, thresholded, min_region, minx, miny)
        new_baseline = np.array(list(Si.keys()))
        new_baseline[:,1] -= minx 
        new_baseline[:,0] -= miny 
        dilated, dist_transform, watershed_segments = separate_lines(thresholded, min_region, height)
        binary_image, polygon_points = merge_line_segments(dilated, new_baseline, watershed_segments)
        filtered_shape = remove_outliers(binary_image, dilated, polygon_points)
        contour = trace_contour(filtered_shape)
        full_size_contour = contour.copy()
        full_size_contour[:,1] += minx 
        full_size_contour[:,0] += miny 
        segmentations.append(full_size_contour.astype(int))


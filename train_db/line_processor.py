import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import grey_closing, grey_dilation, grey_erosion
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.exposure import is_low_contrast


def preprocess_image(orig_image, params):
    '''
    Removes black borders from the image using projection profiles
    '''
    if len(orig_image.shape) == 3:
        image = rgb2gray(orig_image)
    else:
        image = orig_image

    # Binirize the image
    t = threshold_otsu(image)
    image = image > t

    # Image shape 
    Ix = image.shape[1]
    Iy = image.shape[0]

    def get_leftB_coords(I, H, L1, L2):
        filtered_x0 = [x for x in range(int(I/5)) if H[x] > L1 or H[x] < L2] 
        x0 = min(filtered_x0 if len(filtered_x0) > 0 else [-1])
        if x0 == -1:
            return -1

        if H[x0] > L1:
            filtered_x1 = [x for x in range(x0, int(I/2)) if H[x] < L2]
        else:
            filtered_x1 = [x for x in range(x0, int(I/2)) if H[x] > L1]

        x1 = min(filtered_x1 if len(filtered_x1) > 0 else [-1])
        if x1 == -1:
            return -1
        return x1


    def get_rightB_coords(I, H, L1, L2):
        filtered_x0 = [x for x in range(I-1, int(I/5), -1) if H[x] > L1 or H[x] < L2] 
        x0 = max(filtered_x0 if len(filtered_x0) > 0 else [-1])
        if x0 == -1:
            return -1

        if H[x0] > L1:
            filtered_x1 = [x for x in range(x0, int(I/2), -1) if H[x] < L2]
        else:
            filtered_x1 = [x for x in range(x0, int(I/2), -1) if H[x] > L1]

        x1 = max(filtered_x1 if len(filtered_x1) > 0 else [-1])
        if x1 == -1:
            return -1
        return x1

    # Tuning parameters for horizontal crop
    L1 = params['L1_ratio'] * Iy
    L2 = params['L2_ratio'] * Iy

    # Compute vertical histogram
    Hv = np.sum((image==False), axis=0)

    x = get_leftB_coords(Ix, Hv, L1, L2)
    # If there is a black border on the left - crop it.
    xb1 = x if x != -1 else 0
    x = get_rightB_coords(Ix, Hv, L1, L2) 
    # If there is a black border on the right - crop it.
    xb2 = x if x != -1 else Ix

    # Tuning parameters for vertical crop
    L1 = params['L1_ratio'] * Iy
    L2 = params['L2_ratio'] * Iy

    # Horizontal histogram
    Hh = np.sum((image==False)[:,xb1:xb2], axis=1)

    y = get_leftB_coords(Iy, Hh, L1, L2)
    # If there is a black border at the top - crop it.
    yb1 = y if y != -1 else 0
    y = get_rightB_coords(Iy, Hh, L1, L2)
    # If there is a black border at the bottom - crop it.
    yb2 = y if y != -1 else Iy

    image = image[yb1:yb2, xb1:xb2]
    orig_image = orig_image[yb1:yb2, xb1:xb2]
    return (image == False), orig_image

def pad_perimeter(image, comp, padr = 10, padc = 10):
    '''
    Given the image and bounding box, expand bbox such that
    it does not go outside the image
    '''
    r0, r1, c0, c1 = (comp[0], comp[1], comp[2], comp[3])
    r0 = (r0 - padr) if (r0 - padr) > 0 else 0
    r1 = (r1 + padr) if (r1 + padr) < image.shape[0] else image.shape[0] - 1
    c0 = (c0 - padc) if (c0 - padc) > 0 else 0
    c1 = c1 + padc if (c1 + padc) < image.shape[1] else image.shape[1] - 1
    rr, cc = [r0, r0, r1, r1], [c0, c1, c1, c0]
    return rr, cc 

def segment_lines(orig_image, out_path, params):
    '''
    Segments the lines in the given `img_to_read` image and
    saves them in the `lines` directory, which gets created
    in `out_path` dir.
    '''
    
    # returns cropped binary and original images
    image, orig_image = preprocess_image(orig_image, params)

    # Fill in between character spaces
    result = grey_dilation(image, footprint=np.ones((params['first_dilation_rect_h'], 
                                                     params['first_dilation_rect_w'])))
    # Remove ascenders/descenders
    result = grey_erosion(result, footprint=np.ones((params['erosion_rect_h'], 
                                                     params['erosion_rect_w'])))
    # Significantly expand line boxes. This is under assumption that image
    # contains homogeneous text lines.
    transformed = grey_dilation(result, footprint=np.ones((params['second_dilation_rect_h'],
                                                           params['second_dilation_rect_w'])))

    # Works best for EMUNCH dataset:
    # # Aim to close spaces in between characters, form a rectangle like line
    # result = grey_closing(image, footprint=np.ones((params['binary_closing_rect_h'], 
    #                                                 params['binary_closing_rect_w'])))
    # # Aim to remove visible pixel noise. Leave only horizontally long features
    # result = grey_erosion(result, footprint=np.ones(()))
    # # Aim to expand vertical features with dilation
    # transformed = grey_dilation(result, footprint=np.ones((params['binary_dilation_rect_h'],
    #                                                        params['binary_dilation_rect_w'])))

    labeled_image = label(transformed, connectivity=2)
    regions = regionprops(labeled_image)
    components = []

    def get_same_level_comp(mid):
        comp = None
        if len(components) > 0:
            comp = components[-1]
        else:
            return comp

        neighbor_mid = (comp[0] + comp[1]) / 2

        # Compare mid points of neighboring components 
        if np.abs(neighbor_mid - mid) < params['inter_line_spacing']:
            return len(components) - 1
        return None

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        diffr = maxr - minr 
        diffc = maxc - minc
        # Filter line-like regions
        if (diffr < params['line_max_h'] and diffr > params['line_min_h']) and (diffc > params['line_min_w']):
            mid =  (minr+maxr)/2
            neighbor_id = get_same_level_comp(mid)
            if neighbor_id != None:
                prevComp = components[neighbor_id]
                del components[neighbor_id]
                minr_new = min(prevComp[0], minr)
                maxr_new = max(prevComp[1], maxr)
                minc_new = min(prevComp[2], minc)
                maxc_new = max(prevComp[3], maxc)
                combinedBBoxValues = (minr_new, maxr_new, minc_new, maxc_new)
                components.append(combinedBBoxValues)
            else:
                components.append((minr, maxr, minc, maxc))

    images_to_return = []
    for comp in components:
        # Add additional padding to cope with strictly eroded line regions
        rr, cc = pad_perimeter(orig_image, comp, padr = params['bbox_row_pad'], padc = params['bbox_col_pad'])
        line_image = orig_image[rr[0]:rr[2], cc[0]:cc[1]]
        images_to_return.append(line_image)
    return images_to_return
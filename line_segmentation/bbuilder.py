import numpy as np
from scipy.ndimage import binary_erosion, binary_opening
from scipy.spatial import Delaunay
from skimage import img_as_float32

# Constants as defined in Grüning et al. paper:
# "A two-stage method for text line detection in historical documents"
b=0.2
d=10

def skeletonize(Bb):
    '''
    Skeletonize using Lantuéjoul's method.

    Parameters
    ----------
    Bb : array_like
       Binarized image
    '''
    skeleton = np.zeros_like(Bb)
    e = (np.copy(Bb) > 0) * 1
    while e.max()>0:
        o = binary_opening(e)
        skeleton = skeleton | (e & 1-o)
        e = binary_erosion(e)
    return skeleton

def select_superpixels(B, Bs):
    '''
    Filter superpixels in the image.

    Parameters
    ----------
    B : array_like
       Baseline probability image
    Bs : array_like
       Skeletonized baseline image
    '''
    sorted_intensity_indexes = np.argsort(B.ravel())[::-1]
    superpixel_img = np.zeros_like(Bs, dtype=np.uint8)
    superpixels = []
    for index in sorted_intensity_indexes:
        # Convert the flattened index to 2D indices
        i, j = np.unravel_index(index, B.shape)

        if not Bs[i,j]:
            continue

        # If probability is worse than just a guess - pixel should not be part of baseline.
        # Further processing is not needed
        if B[i,j] < b:
            break

        # Keeping number of superpixels small
        a = np.array([i,j])
        isValid = (np.array([np.linalg.norm(a-b) for b in superpixels]) >= 10).all()
        if isValid:
            superpixel_img[i,j] = 255
            superpixels.append(a)
    return superpixels, superpixel_img

def compute_connectivity(e, I):
    '''
    Computes connectivity function - 
    the average intensity of image along the line segment `e`.
    ----------
    Parameters
    ----------
    e : array_like
        e = (p, q), where p and q are arrays of coordinates of point p and q
    I : array_like
        Input image
    '''
    p, q = e

    segment_vector = q - p
    # Discretize the integral
    num_points = int(np.linalg.norm(segment_vector))
    intensity_sum = 0
    for t in range(num_points):
        point = np.round(p + t/num_points * segment_vector).astype(int)
        intensity_sum += I[point[0], point[1]]
    return intensity_sum/num_points


def extract_lto(p, N, B):
    '''
    Extract local text orientation angle for each superpixel
    ----------
    Parameters
    ----------
    p : list_like
        Superpixel coordinates
    T : object_like
        Delaunay tessellation object
    B : array_like
        The baseline prediction image B
    '''
    pointList = N.points.tolist()
    if not isinstance(p, list):
        p = p.tolist()
    p_index = pointList.index(p)
    
    pointer_to_vertex_neighbors, neighbors = N.vertex_neighbor_vertices
    neighbor_vertices = neighbors[pointer_to_vertex_neighbors[p_index]: \
                                  pointer_to_vertex_neighbors[p_index + 1]]
    nv_coords = [N.points[q] for q in neighbor_vertices]

    M = [(np.array(p), q) for q in nv_coords]
    L = np.array([[e, compute_connectivity(e, B)+i] for i,e in enumerate(M)])
    # Sort by baseline connectivities.
    L = L[(-L[:,1]).argsort()]
    if len(L) == 1:
        e = L[0,0]
    else:
        # L[0,0] -> 0th neighbor of p (based on connectivity). Second index denotes
        # edge coordinate position. Next index (1) selects q or r coords that are not p.
        e = (L[0,0][1], L[1,0][1]) 
    return np.arctan(abs(e[1][1] - e[0][1])/abs(e[1][0]-e[0][0]))

def build_baselines(img):
    #call to aru-net
    B = img_as_float32(None)
    Bb = (B > b) * 1
    Bs = skeletonize(Bb)
    S, SI = select_superpixels(B, Bs)
    print(f"{len(S)} superpixels extracted")
    N = Delaunay(S)
    states = [[p, (extract_lto(p, N, B),)] for p in S]



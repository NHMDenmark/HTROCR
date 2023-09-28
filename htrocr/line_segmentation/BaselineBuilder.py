import numpy as np
from scipy.spatial import Delaunay
from skimage.morphology import skeletonize
from htrocr.line_segmentation.predictor.inference import Predictor
from htrocr.line_segmentation.util import get_point_neighbors
import matplotlib.pyplot as plt

class BaselineBuilder():
    def __init__(self, config):
        self.config = config
        self.predictor = Predictor(config)
        

    def __select_superpixels(self, B, Bs):
        '''
        Filter superpixels in the image.

        Parameters
        ----------
        B : array_like
            Baseline probability image
        Bs : array_like
            Skeletonized baseline image

        Returns
        -------
        superpixels : list of ndarray
            A list of superpixel coordinates extracted from baseline prediction
        superpixel_img : list of ndarray
            Binary image of superpixels
        '''
        non_zero_pixels = np.nonzero((B != 0) & (Bs != 0))
        Bpixel_idx = np.column_stack(non_zero_pixels)
        Bfocused = B[Bpixel_idx[:, 0], Bpixel_idx[:, 1]]
        sorted_intensity_indexes = np.argsort(Bfocused, kind='heapsort')[::-1]
        superpixel_img = np.zeros_like(Bs, dtype=np.uint8)
        superpixels = []
        for i in sorted_intensity_indexes:
            a = Bpixel_idx[i]
            if not len(superpixels):
                superpixels.append(a)                
                superpixel_img[tuple(a)]  = 255
            else:
                dists = np.sqrt(((a - np.array(superpixels))**2).sum(axis=1)) 
                if np.min(dists) >= 10:
                    superpixels.append(a)                    
                    superpixel_img[tuple(a)]  = 255        
        return superpixels, superpixel_img

    def __compute_connectivity(self, e, I):
        '''
        Computes connectivity function - 
        the average intensity of image along the line segment `e`.

        Parameters
        ----------
        e : array_like
            e = (p, q), where p and q are arrays of coordinates of point p and q
        I : array_like
            Input image
        
        Returns
        -------
        connectivity : int
            A value showing how many non-zero pixels 
            are within the straight line defined by the given edge 
        '''
        p, q = e

        segment_vector = np.array(q) - np.array(p)
        # Discretize the integral
        num_points = int(np.linalg.norm(segment_vector))
        intensity_sum = 0
        for t in range(num_points):
            point = np.round(p + t/num_points * segment_vector).astype(int)
            intensity_sum += I[point[0], point[1]]
        return intensity_sum/num_points


    def __extract_lto(self, p, N, B):
        '''
        Extract local text orientation angle for each superpixel
        
        Parameters
        ----------
        p : tuple
            Superpixel coordinates
        T : object_like
            Delaunay tessellation object
        B : array_like
            The baseline prediction image B
        
        Returns
        -------
        angle : float
            A local text orientation angle 
        '''
        nv_coords = get_point_neighbors(N, p)
        M = [(np.array(p), q) for q in nv_coords]
        L = np.array([(e, self.__compute_connectivity(e, B)) for e in M], dtype=object)
        # Sort by baseline connectivities.
        L = L[(-L[:,1]).argsort()]
        # do not consider non-neighbor edges
        # max_val = max(L[:,1])
        max_val = L[0,1]
        # L = np.array([l for l in L if l[1]/max_val > 0.4])
        L = L[L[:, 1]/max_val > self.config["neighbour_connectivity_ratio"]]
        if len(L) == 1:
            e = L[0,0]
        else:
            # L[0,0] -> 0th neighbor of p (based on connectivity). Second index denotes
            # edge coordinate position. Next index (1) selects q or r coords that are not p.
            e = (L[0,0][1], L[1,0][1]) 
        # return np.arctan(abs(e[1][0] - e[0][0])/(abs(e[1][1] - e[0][1])+ np.finfo(np.float64).eps))
        qx = e[0][1]
        rx = e[1][1]
        qy = e[0][0]
        ry = e[1][0]
        if qx > rx:
            angle = np.arctan2((-1)*(qy-ry), qx-rx)
	    
        else:
            angle = np.arctan2((-1)*(qy-ry), rx-qx)
        	    	
        return angle

    def __find_projections(self, p, points_within_circle, angle):
        '''
        Finds projections of given points on the orientation vector 
        '''
        sp_angle = angle+(np.pi/2)
        cos = np.cos(sp_angle)
        sin = np.sin(sp_angle)
        orientation_vector = np.array([cos, sin])
        pvectors = np.array([points_within_circle[:,1] - p[1], (-1) * (points_within_circle[:,0] - p[0])])
        angles = np.arctan2(pvectors[1], pvectors[0])
        valid_area_point_idx = []
        for i, t in enumerate(angles):
            if (sp_angle + np.pi/6) > t and (sp_angle - np.pi/6) < t:
                valid_area_point_idx.append(i)
        valid_area_points = points_within_circle[valid_area_point_idx]
        if len(valid_area_points) == 0:
            return -1
        pq = np.stack((valid_area_points[:,1] - p[1], -(valid_area_points[:,0]-p[0])), axis=-1)
        projections = np.dot(pq, orientation_vector)
        distance_to_pline = np.abs(cos*(p[0]-valid_area_points[:,0]) - sin*(p[1]-valid_area_points[:,1]))
        sums = projections + distance_to_pline
        closest_point2line_idx = np.argmin(sums)
        projection = projections[closest_point2line_idx]
        return int(projection)

    def __select_closest_interpdist_within_circle(self, p, angle, points):
        '''
        Finds interline distances
        '''
        rs = np.array([32, 64, 128, 256])
        points = np.array([a for a in points if p[0] != a[0] and p[1] != a[1]])
        for radius in rs:
            distances = np.linalg.norm(abs(points - p), axis=1)
            mask = distances <= radius
            points_within_circle = points[mask]
            projection = self.__find_projections(p, points_within_circle, angle)
            if projection != -1:
                break
        if projection == -1:
            return self.config["fixed_interline_dist"]
        return projection

    def __L2_norm(self, p, q):
        return np.sqrt((q[1] - p[1])**2 + (q[0] - p[0])**2)

    def __find_cluster_index(self, p, P_star):
        '''
        Search for cluster where the point p belongs
        '''
        for i, S_i in enumerate(P_star):
            if p in S_i:
                return i
        return -1

    def __remove_cfcc(self, Sx, S0):
        '''
        Remove cluster from clutter cluster
        '''
        for k in Sx:
            if k in S0:
                del S0[k]
        return S0

    def __remove_cfpl(self, Sx, P):
        '''
        Remove cluster from partition list
        '''
        return [Si for Si in P if Si != Sx]

    def __cluster_points(self, states, Bs, N):
        S0 = states.copy()
        P_star = [S0] 
        for p, v in states.items():
            # print(p, v)
            NVs = get_point_neighbors(N, p)
            nv_connectivity = []
            p_state = {p: v}
            for nv in NVs:
                q = tuple(nv.astype(int))
                isSameOriented = (abs(states[p][0] - states[q][0]) % np.pi) <= np.pi/6
                interpdist = self.__L2_norm(p, q)
                # if not following the skeleton - skip
                if isSameOriented and interpdist < 25:
                    nv_connectivity.append(q)
            if len(nv_connectivity) == 0:
                continue
            for q in nv_connectivity:
                p_idx = self.__find_cluster_index(p, P_star)
                q_idx = self.__find_cluster_index(q, P_star)
                q_state = {q: states[q]}

                if p_idx == 0 and q_idx == 0:
                    # Create a new cluster
                    union_state = p_state.copy()
                    union_state.update(q_state)
                    Snew = union_state
                    S0 = self.__remove_cfcc(Snew, S0)
                    P_star.append(Snew)
                elif p_idx == 0 and q_idx > 0:
                    # Merge p to a cluster
                    set_union = P_star[q_idx].copy()
                    set_union.update(p_state)
                    P_star[q_idx] = set_union
                    S0 = self.__remove_cfcc(p_state, S0)
                elif q_idx == 0 and p_idx > 0:
                    # Merge q to a cluster
                    set_union = P_star[p_idx].copy()
                    set_union.update(q_state)
                    P_star[p_idx] = set_union
                    S0 = self.__remove_cfcc(q_state, S0)
                elif p_idx != q_idx and (p_idx > 0 and q_idx > 0):
                    # Attempt to merge p and q neighboring clusters
                    Si = P_star[p_idx].copy()
                    Sj = P_star[q_idx].copy()
                    union_set = Si.copy()
                    union_set.update(Sj)
                    P_star[p_idx] = union_set
                    P_star = self.__remove_cfpl(Sj, P_star)
                # Otherwise - ignore since both points are in the same cluster
        # Remove clutter cluster from segmentations
        P_star = P_star[1:]
        # Remove possible plant noise that contan less than 5 pixels in the baseline
        P_star = [set_i for set_i in P_star if len(set_i) >= 5]
        return P_star

    def run(self, img):
        
        B = self.predictor.run(img)
        Bb = (B > self.config['superpixel_confidence_thresh']) * 1
        Bs = skeletonize(Bb)
        S, SI = self.__select_superpixels(B, Bs)
        if len(S) == 0 or len(S) < 3:
            return [], None
        N = Delaunay(S)
        angles = [self.__extract_lto(p, N, B) for p in S]
        states = {tuple(p): (angles[i], self.__select_closest_interpdist_within_circle(p, angles[i], S)) for i, p in enumerate(S)}
        
        clusters = self.__cluster_points(states, Bs, N)
        
        print(f"{len(S)} superpixels extracted; {len(clusters)} line clusters generated")
        return clusters, N
    
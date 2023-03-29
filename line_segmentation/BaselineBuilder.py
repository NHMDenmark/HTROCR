import numpy as np
from scipy.ndimage import binary_erosion, binary_opening
from scipy.spatial import Delaunay
from arunet import Inference_pb

class BaselineBuilder():
    def __init__(self, config):
        self.params = config
        self.predictor = Inference_pb(config['arunet_weight_path'])

    def __skeletonize(self, Bb):
        '''
        Skeletonize using Lantuéjoul's method.

        Parameters
        ----------
        Bb : array_like
            Binarized image
        
        Returns
        -------
        skeleton : ndarray of bools
            Eroded skeleton of the given input.
        '''
        skeleton = np.zeros_like(Bb, dtype=np.uint8)
        while Bb.max()>0:
            o = binary_opening(Bb)
            skeleton = skeleton | (Bb & 1-o)
            Bb = binary_erosion(Bb)
        return skeleton

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
            if B[i,j] < self.config['superpixel_confidence_thresh']:
                break

            # Keeping number of superpixels small
            a = np.array([i,j])
            isValid = (np.array([np.linalg.norm(a-b) for b in superpixels]) >= 10).all()
            if isValid:
                superpixel_img[i,j] = 255
                superpixels.append(a)
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
        skeleton : ndarray of bools
            Eroded skeleton of the given input.
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

    def __get_point_neighbours(self, N, p):
        '''
        Returns all neighbours for point in the neighbourhood system

        Parameters
        ----------
        N : object_like
            Delaunay triangulation object 
        p : array_like
            point coords
        
        Returns
        -------
        nv_coords : list of ndarray
            Neighbouring vertices
        '''
        pointList = N.points.tolist()
        if not isinstance(p, list):
            p = list(p)
        p_index = pointList.index(p)
        pointer_to_vertex_neighbors, neighbours = N.vertex_neighbor_vertices
        neighbour_vertices = neighbours[pointer_to_vertex_neighbors[p_index]: \
                                    pointer_to_vertex_neighbors[p_index + 1]]
        nv_coords = [N.points[q] for q in neighbour_vertices]
        return nv_coords

    def __extract_lto(self, p, N, B):
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
        nv_coords = self.get_point_neighbors(N, p)
        M = [(np.array(p), q) for q in nv_coords]
        L = np.array([(e, self.__compute_connectivity(e, B)) for e in M], dtype=object)
        # Sort by baseline connectivities.
        L = L[(-L[:,1]).argsort()]
        # do not consider non-neighbor edges
        max_val = max(L[:,1])
        L = np.array([l for l in L if l[1]/max_val > 0.4])
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
            angle = np.arctan2((-1)*(qy-ry), qx-rx) #if qy > ry else np.arctan2(ry-qy, qx-rx)
        else:
            angle = np.arctan2((-1)*(qy-ry), rx-qx) #if qy > ry else np.arctan2(ry-qy, rx-qx)
        return angle

    def __find_projections(self, p, points_within_circle, angle):
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
        distance_to_pline = np.abs(cos*(p[0]-valid_area_points[:,0]) - sin*(p[1]-valid_area_points[:,1]))
        closest_point2line_idx = np.argmin(distance_to_pline)
        pproject = valid_area_points[closest_point2line_idx]
        pq = [pproject[1] - p[1], (-1)*(pproject[0]-p[0])]
        projection = np.dot(pq, orientation_vector)
        return int(projection)

    def __select_closest_interpdist_within_circle(self, p, angle, points):
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
            return 120
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

    def __small_distance(self, S1, S2):
        for pk in S1.keys():
            for qk in S2.keys():
                if self.__L2_norm(pk, qk) < 20:
                    return True

    def __cluster_points(self, states, Bs):
        S0 = states.copy()
        P_star = [S0] 
        unconnected_ps = []
        for p in states:
            NVs = self.__get_point_neighbours(N, p)
            nv_connectivity = []
            p_state = {p: states[p]}
            for nv in NVs:
                q = tuple(nv.astype(int))
                isSameOriented = (abs(states[p][0] - states[q][0]) % np.pi) <= np.pi/4
                interpdist = self.__L2_norm(p, q)
                # if not following the skeleton - skip
                if isSameOriented and interpdist < 20:
                    nv_connectivity.append([q, self.__compute_connectivity((p,q), Bs)])
            if len(nv_connectivity) == 0:
                unconnected_ps.append(p)
                continue
            strongest_NVs = np.array(nv_connectivity)
            strongest_NVs[::-1,1].sort()
            if len(strongest_NVs) > 1 and strongest_NVs[1,1]/(strongest_NVs[0,1]+np.finfo(float).eps) > 0.3:
                strongest_NVs = strongest_NVs[0:2, 0]
            else:
                strongest_NVs = [strongest_NVs[0, 0]]

            for q in strongest_NVs:
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

        # Fix connectivity mistakes in skeleton - if distance between segements is no larger than the discretization upperlimit - the clusters should be merged. Otherwise it is a mistake of ARU-Net
        n = len(P_star)
        i = 1
        while i < n:
            j=i+1
            wasMerged = False
            while j < n:
                if self.__small_distance(P_star[i], P_star[j]):
                    P_star[i] |= P_star[j]
                    del P_star[j]
                    n -= 1
                    wasMerged = True
                else:
                    j += 1
            if not wasMerged:
                i += 1
        P_star = [set_i for set_i in P_star if len(set_i) >= 5]
        return P_star

    def run(self, img):
        B = self.predictor.run(img)
        Bb = (B > 0.2) * 1
        Bs = self.__skeletonize(Bb)
        S, SI = self.__select_superpixels(B, Bs)
        print(f"{len(S)} superpixels extracted")
        N = Delaunay(S)
        angles = [self.__extract_lto(p, N, B) for p in S]
        states = {tuple(p): (angles[i], self.__select_closest_interpdist_within_circle(p, angles[i], S)) for i, p in enumerate(S)}
        clusters = self.__cluster_points(states, Bs)
        print(f"{len(clusters)} line clusters generated")
        return clusters, N
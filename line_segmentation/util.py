def get_point_neighbors(N, p):
    '''
    Returns all neighbours for point in the neighbourhood system

    Parameters
    ----------
    N : object_like
        Delaunay triangulation object 
    p : tuple
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
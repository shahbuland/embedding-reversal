import umap
import numpy as np

def umap_reduce(vectors):
    reducer = umap.UMAP()
    transformed_points = reducer.fit_transform(vectors)
    return transformed_points

class umap_invert(point, points, embeddings, nearest_neighbors = 10):
    """
    Inverts UMAP for a single point

    :param point: Point to get embedding for
    :param points: Points corresponding to all embeddings
    :param embeddings: Embeddings original
    :param nearest_neighbors: How many NN to look at
    """

    dists = np.linalg.norm(points - point, axis=1)
    idx = np.argpartition(dists, nearest_neighbors)[:nearest_neighbors]

    # embeddings is [n, 4096], [idx] should be [nearest_neighbors, 4096]
    return embeddings[idx].mean(0) # this should be [4096]


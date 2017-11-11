from sklearn.neighbors import  NearestNeighbors
import numpy as np
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x)
distances, indices = nbrs.kneighbors(x)
print(distances)

print("-----")

print(indices)
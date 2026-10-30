import time

from aspartik.data.tree import (
    read_binary_trees_path,
    # robinson_foulds_matrix,
    triplet_distance_matrix,
)

start = time.perf_counter()
trees = read_binary_trees_path("run-full.trees")
end = time.perf_counter()
print(f"read: {end - start:.2f}sec")

start = time.perf_counter()
m = triplet_distance_matrix(trees[:100], num_threads=16)
end = time.perf_counter()
print(f"triplet: {end - start:.2f}sec")

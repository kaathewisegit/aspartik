from .._aspartik_rust_impl._data_rust_impl import BinaryTree as BinaryTree, Tree as Tree


def classical_mds(distances, dimensions: int = 2, *, correction: bool = False):
    import numpy as np

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if not np.isfinite(distances).all():
        raise ValueError("distances must be finite")
    if (distances < 0).any():
        raise ValueError("distances must be nonnegative")
    if not np.allclose(distances, distances.T):
        raise ValueError("distances must be symmetric")
    if not np.allclose(np.diag(distances), 0.0):
        raise ValueError("the distance matrix diagonal must be zero")

    size = distances.shape[0]
    if size == 0:
        return np.empty((0, dimensions), dtype=np.float64)
    centering = np.eye(size) - np.full((size, size), 1.0 / size)

    def gram_matrix(values):
        return -0.5 * centering @ np.square(values) @ centering

    gram = gram_matrix(distances)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if correction and eigenvalues[0] < 0.0:
        squared = np.square(distances)
        squared += 2.0 * -eigenvalues[0]
        np.fill_diagonal(squared, 0.0)
        gram = -0.5 * centering @ squared @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1]
    selected = order[: min(dimensions, size)]
    coordinates = eigenvectors[:, selected] * np.sqrt(
        np.maximum(eigenvalues[selected], 0.0)
    )
    if coordinates.shape[1] < dimensions:
        coordinates = np.pad(
            coordinates, ((0, 0), (0, dimensions - coordinates.shape[1]))
        )
    return coordinates


def plot_tree_space(
    fig,
    ax,
    distances,
    *,
    chain_labels=None,
    posterior=None,
    correction: bool = False,
):
    import numpy as np

    if chain_labels is not None and posterior is not None:
        raise ValueError("chain_labels and posterior cannot be used together")
    coordinates = classical_mds(distances, correction=correction)
    if coordinates.shape[0] == 0:
        return coordinates
    if chain_labels is not None:
        labels = np.asarray(chain_labels)
        if labels.shape != (coordinates.shape[0],):
            raise ValueError("chain_labels must have one value per tree")
        for label in dict.fromkeys(labels.tolist()):
            selected = labels == label
            ax.scatter(coordinates[selected, 0], coordinates[selected, 1], label=label)
    elif posterior is not None:
        posterior = np.asarray(posterior, dtype=np.float64)
        if posterior.shape != (coordinates.shape[0],):
            raise ValueError("posterior must have one value per tree")
        points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=posterior)
        fig.colorbar(points, ax=ax)
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1])
    return coordinates

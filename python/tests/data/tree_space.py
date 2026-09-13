import numpy as np
import pytest
from matplotlib import pyplot as plt

from aspartik.data.tree import classical_mds, plot_tree_space

plt.switch_backend("Agg")


def pairwise_distances(points):
    differences = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.square(differences).sum(axis=2))


def test_classical_mds_reconstructs_euclidean_distances():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
    distances = pairwise_distances(points)
    coordinates = classical_mds(distances)

    assert coordinates.shape == (4, 2)
    assert np.allclose(coordinates.mean(axis=0), 0.0)
    assert np.allclose(pairwise_distances(coordinates), distances)


def test_classical_mds_lingoes_correction():
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    coordinates = classical_mds(distances, dimensions=3, correction=True)

    assert coordinates.shape == (4, 3)
    assert np.isfinite(coordinates).all()
    assert np.allclose(coordinates.mean(axis=0), 0.0)


@pytest.mark.parametrize(
    ("distances", "message"),
    [
        ([[0.0, 1.0]], "square"),
        ([[0.0, -1.0], [-1.0, 0.0]], "nonnegative"),
        ([[0.0, 1.0], [2.0, 0.0]], "symmetric"),
        ([[1.0, 0.0], [0.0, 0.0]], "diagonal"),
        ([[0.0, np.inf], [np.inf, 0.0]], "finite"),
    ],
)
def test_classical_mds_rejects_invalid_matrices(distances, message):
    with pytest.raises(ValueError, match=message):
        classical_mds(distances)


def test_classical_mds_empty_and_dimensions():
    assert classical_mds(np.empty((0, 0)), dimensions=4).shape == (0, 4)
    with pytest.raises(ValueError, match="positive"):
        classical_mds(np.zeros((1, 1)), dimensions=0)


def test_plot_tree_space_with_chain_labels():
    distances = np.array(
        [
            [0.0, 1.0, 10.0, 10.0],
            [1.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 1.0],
            [10.0, 10.0, 1.0, 0.0],
        ]
    )
    fig, ax = plt.subplots()
    coordinates = plot_tree_space(
        fig, ax, distances, chain_labels=["first", "first", "second", "second"]
    )

    first_center = coordinates[:2].mean(axis=0)
    second_center = coordinates[2:].mean(axis=0)
    assert np.linalg.norm(first_center - second_center) > 5.0
    assert len(ax.collections) == 2
    assert [item.get_label() for item in ax.collections] == ["first", "second"]
    plt.close(fig)


def test_plot_tree_space_with_posterior():
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    fig, ax = plt.subplots()
    coordinates = plot_tree_space(fig, ax, distances, posterior=[-10.0, -5.0])

    assert coordinates.shape == (2, 2)
    assert len(ax.collections) == 1
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_tree_space_rejects_bad_colors():
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="cannot be used together"):
        plot_tree_space(
            fig,
            ax,
            distances,
            chain_labels=["a", "b"],
            posterior=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="one value per tree"):
        plot_tree_space(fig, ax, distances, chain_labels=["a"])
    with pytest.raises(ValueError, match="one value per tree"):
        plot_tree_space(fig, ax, distances, posterior=[0.0])
    plt.close(fig)

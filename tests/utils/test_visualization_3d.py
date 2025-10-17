import os

import numpy as np
import pytest
import pyvista as pv
from pyscf import gto

from mldft.utils.visualize_3d import plot_molecule, visualize_density, visualize_orbital

mols_image_sizes = [(gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"), (200, 200))]

headless = (
    os.name != "posix"  # display check probably only works on linux
    or "DISPLAY" not in os.environ
    or not os.environ["DISPLAY"]
    or os.environ["DISPLAY"].startswith("localhost")
)


@pytest.mark.skipif(headless, reason="Cannot confirm display available")
@pytest.mark.parametrize("mol, image_size", mols_image_sizes)
def test_plot_molecule(mol, image_size):
    """Test whether plot_molecule produces images."""
    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    plot_molecule(mol, plotter=pl)
    image = pl.show(screenshot=True)

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"


mols_image_sizes_vis_kwargs = [
    (gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"), (100, 100), dict(mode="volume")),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="volume", plot_molecule="false"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="nested_isosurfaces"),
    ),
]


@pytest.mark.skipif(headless, reason="Cannot confirm display available")
@pytest.mark.parametrize("mol, image_size, visualize_orbital_kwargs", mols_image_sizes_vis_kwargs)
def test_visualize_orbital(mol, image_size, visualize_orbital_kwargs):
    """Test whether visualize_orbital produces images."""
    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    visualize_orbital(mol, coeff=np.ones(mol.nao), plotter=pl, **visualize_orbital_kwargs)
    image = pl.show(screenshot=True)

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"


mols_image_sizes_vis_kwargs = [
    # (gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"), (100, 100), dict(mode='volume')),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface", plot_molecule=False),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="nested_isosurfaces"),
    ),
]


@pytest.mark.skipif(headless, reason="Cannot confirm display available")
@pytest.mark.parametrize("mol, image_size, visualize_density_kwargs", mols_image_sizes_vis_kwargs)
def test_visualize_density(
    mol: gto.Mole, image_size: tuple[int, int], visualize_density_kwargs: dict
):
    """Test whether visualize_density produces images."""
    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    visualize_density(mol, density_matrix=np.eye(mol.nao), plotter=pl, **visualize_density_kwargs)
    image = pl.show(screenshot=True)

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"

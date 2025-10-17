from functools import partial

import numpy as np
import pyscf
import pyvista as pv
from matplotlib.pyplot import rcParams
from pyscf import dft

from mldft.utils.conversions import pyscf_to_rdkit
from mldft.utils.cube_files import DataCube


def find_isosurface_value(
    cube_array: np.ndarray, quantile: float | np.ndarray = 0.9, p: int = 2
) -> float | np.ndarray:
    """Find an isosurface value for a cube array, such that the isosurface contains a given
    fraction of the total mass. The mass is computed as the sum of the absolute values of the cube
    array raised to the power p.

    Args:
        cube_array: The cube array.
        quantile: The fraction (or array of fractions) of the total mass to be contained in the isosurface.
        p: The power to raise the cube array to. Use p=1 for electron density and p=2 for orbitals.

    Returns:
        The isosurface value.
    """
    cube_array = np.abs(cube_array).flatten()
    mass = cube_array**p
    total_mass = mass.sum()
    mass_sorted = np.sort(mass)[::-1]
    ind = np.searchsorted(np.cumsum(mass_sorted), total_mass * quantile)
    isosurface_value = mass_sorted[ind] ** (1 / p)

    return isosurface_value


def _fix_cube_data(cube_data: str) -> str:
    """Fix a cube file string produced by pyscf. Undoes a pyscf bug by adding spaces where needed:
    e.g. replace incorrect lines like
    "-2.52899E-114-1.81539E-116-9.75640E-119-3.92559E-121-1.18255E-123-2.66702E-126" with
    "-2.52899E-114 -1.81539E-116 -9.75640E-119 -3.92559E-121 -1.18255E-123 -2.66702E-126".

    Args:
        cube_data: The cube data to fix.

    Returns:
        The fixed cube data.
    """
    cube_data = cube_data.replace("-", " -")
    cube_data = cube_data.replace("E -", "E-")

    return cube_data


def _eval_orbital(coords: np.ndarray, mol: pyscf.gto.Mole, coeff: np.ndarray) -> np.ndarray:
    """Evaluate an orbital at the given coordinates.

    Args:
        coords: The coordinates to evaluate the orbital at, shape (n, 3).
        mol: The molecule.

    Returns:
        The orbital at the given coordinates, shape (n).
    """
    ao = mol.eval_gto("GTOval", coords)
    orb_on_grid = np.einsum("...i,i", ao, coeff)
    return orb_on_grid


def visualize_orbital(
    mol: pyscf.gto.Mole,
    coeff: np.ndarray,
    resolution: float = 0.3,
    margin: float = 3.0,
    **plot_orbital_kwargs,
):
    """Visualize an orbital together with the molecule it belongs to.

    Args:
        mol: The molecule.
        coeff: The orbital coefficients.
        resolution: The resolution of the cube file.
        margin: The margin of the cube file.
        plot_orbital_kwargs: Keyword arguments to pass to plot_orbital.

    Returns:
        The pyvista plotter.
    """
    cube = DataCube.from_function(
        mol=mol,
        func=partial(_eval_orbital, coeff=coeff, mol=mol),
        resolution=resolution,
        margin=margin,
    )

    return plot_orbital(cube, **plot_orbital_kwargs)


def _eval_density(coords: np.ndarray, mol: pyscf.gto.Mole, dm: np.ndarray) -> np.ndarray:
    """Evaluate the electron density at the given coordinates.

    Args:
        coords: The coordinates to evaluate the density at, shape (n, 3).
        mol: The molecule.
        dm: The density matrix.

    Returns:
        The electron density at the given coordinates, shape (n).
    """
    ao = mol.eval_gto("GTOval", coords)
    rho = dft.numint.eval_rho(mol, ao, dm)
    return rho


def visualize_density(
    mol: pyscf.gto.Mole,
    density_matrix: np.ndarray,
    resolution: float = 0.3,
    margin: float = 3.0,
    **plot_density_kwargs,
):
    """Visualize the electron density with the corresponding molecule.

    Args:
        mol: The molecule.
        density_matrix: The density matrix in the atomic orbital basis.
        resolution: The resolution of the cube file.
        margin: The margin of the cube file.
        plot_density_kwargs: Keyword arguments to pass to plot_density.

    Returns:
        The pyvista plotter.
    """

    cube = DataCube.from_function(
        mol=mol,
        func=partial(_eval_density, dm=density_matrix, mol=mol),
        resolution=resolution,
        margin=margin,
    )

    return plot_density(cube, **plot_density_kwargs)


ATOM_COLORS = {
    "C": "#c8c8c8",
    "H": "#ffffff",
    "N": "#8f8fff",
    "S": "#ffc832",
    "O": "#f00000",
    "F": "#ffff00",
    "P": "#ffa500",
    "K": "#42f4ee",
    "G": "#3f3f3f",
}

ATOM_IDS = {key: i for i, key in enumerate(ATOM_COLORS.keys())}


def get_sticks_mesh_dict(
    mol: pyscf.gto.Mole,
    bond_radius: float = 0.2,
    atom_radius: float = None,
    resolution=20,
) -> dict:
    """Create a 'sticks' representation for a pyscf molecule. By choosing a larger `atom_radius`,
    one can make this look like balls-and-sticks.

    Args:
        mol: The molecule.
        bond_radius: The radius of the cylinders representing the bonds.
        atom_radius: The radius of the spheres representing the atoms. If None, use `bond_radius`.
        resolution: The resolution of the cylinders and spheres.

    Returns:
        A dictionary with keyword arguments to pass to pyvista.Plotter.add_mesh.
    """
    rdkit_mol = pyscf_to_rdkit(mol)
    atom_positions = (
        rdkit_mol.GetConformer().GetPositions() * 1.8897259886
    )  # rdkit gives values in Angstrom

    atom_radius = bond_radius if atom_radius is None else atom_radius

    mesh_elements = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        mesh_elements_atom = []
        pos_i = atom_positions[i]
        color_id = ATOM_IDS.get(atom.GetSymbol(), ATOM_IDS["G"])
        sphere = pv.Sphere(
            center=pos_i,
            radius=atom_radius,
            phi_resolution=resolution,
            theta_resolution=resolution,
        )
        sphere["color_ids"] = np.ones(sphere.n_cells) * color_id
        mesh_elements_atom.append(sphere)

        bonds = atom.GetBonds()
        for bond in bonds:
            inds = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            j = inds[0] if inds[1] == i else inds[1]  # select the index of the other atom
            pos_j = atom_positions[j]
            cylinder = pv.Cylinder(
                center=(pos_i * 3 + pos_j) / 4,  # one quarter of the way from i to j
                direction=pos_j - pos_i,
                radius=bond_radius,
                height=np.linalg.norm(pos_i - pos_j) / 2,
                capping=False,
                resolution=resolution,
            )
            cylinder["color_ids"] = np.ones(cylinder.n_cells) * color_id
            mesh_elements_atom.append(cylinder)

        mesh_elements.extend(mesh_elements_atom)

    merged_mesh = pv.MultiBlock(mesh_elements).combine().extract_surface()
    add_mesh_kwargs = dict(
        mesh=merged_mesh,
        smooth_shading=True,
        diffuse=0.5,
        specular=0.5,
        ambient=0.5,
        clim=(0, len(ATOM_COLORS)),
        cmap=list(ATOM_COLORS.values()),
        show_scalar_bar=False,
    )
    return add_mesh_kwargs


# colors of the three axes in the local frame visualization
AXES_COLORS = ["#ff0000", "#00ff00", "#0000ff"]


def get_local_frames_mesh_dict(
    origins: np.ndarray,
    bases: np.ndarray,
    scale: float = 1,
    axes_radius_scale: float = 0.03,
    cone_scale: float = 2.0,
    cone_aspect_ratio: float = 2,
    resolution: int = 20,
) -> dict:
    """Create a mesh dict for a set of local frames, given by their origin and basis vectors.

    Args:
        origins: The origins of the local frames, shape (n, 3).
        bases: The basis vectors of the local frames, shape (n, 3=n_vectors, 3=n_axes).
        scale: The scale of the local frames.
        axes_radius_scale: The radius of the cylinders representing the axes.
        cone_scale: The scale of the cones representing the axes.
        cone_aspect_ratio: The aspect ratio of the cones representing the axes.
        resolution: The resolution of the cylinders and cones.

    Returns:
        A dictionary with keyword arguments to pass to :meth:`pyvista.Plotter.add_mesh`.
    """

    assert (
        bases.shape[2] == 3 and 0 < bases.shape[1] <= 3
    ), f"Invalid shape of bases: {bases.shape}. Must be (n, 1|2|3, 3)."
    assert origins.shape[1] == 3, f"Invalid shape of origins: {origins.shape}. Must be (n, 3)."
    assert origins.shape[0] == bases.shape[0], (
        f"Incompatible shapes of origins and bases: {origins.shape} and {bases.shape}. "
        f"Must be (n, 3) and (n, 1|2|3, 3)."
    )

    mesh_elements = []

    axes_radius = scale * axes_radius_scale
    cone_radius = axes_radius * cone_scale
    cone_height = cone_radius * cone_aspect_ratio * 2

    for origin, bases in zip(origins, bases):
        for color_id, basis_vector in enumerate(bases):
            cylinder_height = scale * np.linalg.norm(basis_vector, axis=-1) - cone_height
            if cylinder_height < 0:
                cylinder_height = 0
            basis_vector_length = np.linalg.norm(basis_vector)
            basis_vector = basis_vector / basis_vector_length
            dest = origin + basis_vector * cylinder_height

            cylinder = pv.Cylinder(
                center=(origin + dest) / 2,  # one quarter of the way from i to j
                direction=dest - origin,
                radius=axes_radius,
                height=cylinder_height,
                resolution=resolution,
            )
            cylinder["color_ids"] = np.ones(cylinder.n_points) * color_id
            mesh_elements.append(cylinder)

            # add a cone at the tip of the arrow
            this_cone_height = min(cone_height, basis_vector_length)
            cone = pv.Cone(
                center=origin + basis_vector * (cylinder_height + this_cone_height / 2),
                direction=basis_vector,
                height=this_cone_height,
                radius=cone_radius,
                resolution=resolution,
            )
            cone = cone.extract_geometry()
            cone["color_ids"] = np.ones(cone.n_points) * color_id
            mesh_elements.append(cone)

    merged_mesh = pv.MultiBlock(mesh_elements).combine().extract_surface()
    add_mesh_kwargs = dict(
        mesh=merged_mesh,
        smooth_shading=True,
        diffuse=0.5,
        specular=0.5,
        ambient=0.5,
        clim=(0, len(AXES_COLORS) - 1),
        cmap=AXES_COLORS,
        show_scalar_bar=False,
    )
    return add_mesh_kwargs


def plot_molecule(
    mol: pyscf.gto.Mole,
    plotter: pv.Plotter = None,
    figsize: tuple[int, int] = None,
    title: str = None,
) -> pv.Plotter:
    """Plot molecule.

    Args:
        mol: The pyscf molecule.
        plotter: A pyvista plotter to use. If None, a new plotter is created.
        figsize: The figure size in inches.
        title: The title of the plot.

    Returns:
        The pyvista plotter.
    """

    if plotter is not None:
        pl = plotter
    else:
        plotter_kwargs = dict(image_scale=2)
        if figsize is not None:
            dpi = int(rcParams["figure.dpi"])
            plotter_kwargs["window_size"] = [s * dpi for s in figsize]
        pl = pv.Plotter(**plotter_kwargs)

    pl.add_mesh(**get_sticks_mesh_dict(mol))

    if title is not None:
        pl.add_title(title)

    pl.camera_position = "iso"

    if plotter is None:  # if no plotter was passed, show the plot
        return pl.show()
    else:  # otherwise, return the plotter, e.g. to save the image next
        return pl


def plot_orbital(
    cube: DataCube,
    mode: str = "auto",
    plot_molecule: bool = True,
    isosurface_quantile: float = None,
    plotter: pv.Plotter = None,
    figsize: tuple[int, int] = None,
    title: str = None,
) -> pv.Plotter:
    """Plot an electron orbital using pyvista. By default, the orbital is plotted as a volume.

    Args:
        cube: The `DataCube`.
        mode: The mode to use for plotting. Must be one of 'volume', 'isosurface', 'nested_isosurfaces'.
        plot_molecule: Whether to plot the molecule.
        isosurface_quantile: The quantile of the total mass to be contained in the isosurface, for mode 'isosurface'.
            Defaults to 0.9.
        plotter: A pyvista plotter to use. If None, a new plotter is created.
        figsize: The figure size in inches.
        title: The title of the plot.

    Returns:
        The pyvista plotter.
    """

    if mode == "auto":
        if isosurface_quantile is None:
            mode = "volume"
        else:
            mode = "isosurface"
    assert mode in [
        "volume",
        "isosurface",
        "nested_isosurfaces",
    ], f"Invalid mode: {mode}. Must be one of 'volume', 'isosurface', 'nested_isosurfaces'."

    if plotter is not None:
        pl = plotter
    else:
        plotter_kwargs = dict(image_scale=2)
        if figsize is not None:
            dpi = int(rcParams["figure.dpi"])
            plotter_kwargs["window_size"] = [s * dpi for s in figsize]
        pl = pv.Plotter(**plotter_kwargs)

    orbital = cube.to_pyvista()

    if mode == "volume":
        neg_mask = orbital["data"] < 0
        rgba = np.zeros((orbital.n_points, 4), np.uint8)
        rgba[neg_mask, 0] = 170
        rgba[~neg_mask, 2] = 170

        # normalize opacity, such that 0.1 quantile is fully opaque
        opac = np.abs(orbital["data"])  # ** 2
        opac /= find_isosurface_value(opac, 0.25, p=1)
        opac = np.clip(opac, 0, 1)

        rgba[:, -1] = opac * 255

        orbital["plot_scalars"] = rgba

        vol = pl.add_volume(
            orbital,
            opacity_unit_distance=10,
            scalars="plot_scalars",
        )

        vol.prop.interpolation_type = "linear"

    if plot_molecule:
        mol = cube.mol
        pl.add_mesh(**get_sticks_mesh_dict(mol))

    if mode in ["isosurface", "nested_isosurfaces"]:
        if isinstance(isosurface_quantile, float):
            assert (
                mode == "isosurface"
            ), f'Invalid mode: {mode}. Must be "isosurface" for a single isosurface value'
            quantiles = np.array([isosurface_quantile])
        elif mode == "nested_isosurfaces":
            quantiles = np.linspace(0.1, 0.99, 10)
        else:
            isosurface_quantile = 0.9
            quantiles = np.array([isosurface_quantile])

        isosurface_values = find_isosurface_value(orbital["data"], quantiles, p=2)

        # add negative isosurface values
        isosurface_values = np.concatenate([-isosurface_values, isosurface_values])
        quantiles = np.concatenate([-quantiles, quantiles])

        isosurface = orbital.contour(isosurfaces=isosurface_values)
        iso_mesh = isosurface.extract_geometry()

        iso_mesh["quantile"] = np.zeros(iso_mesh.n_points)
        iso_mesh["opacity"] = np.zeros(iso_mesh.n_points)

        for quantile, isosurface_value in zip(quantiles, isosurface_values):
            mask = iso_mesh["data"] == isosurface_value

            # squaring looks good with 'seismic' colormap
            iso_mesh["quantile"][mask] = quantile
            iso_mesh["opacity"][mask] = 1 - np.abs(quantile)

        if mode == "nested_isosurfaces":
            # add a set of nested, transparent isosurfaces
            iso_mesh["quantile"] = (
                -np.sign(iso_mesh["quantile"]) * (1 - np.abs(iso_mesh["quantile"])) ** 2
            )
            pl.add_mesh(
                iso_mesh,
                scalars="quantile",
                clim=(-1, 1),
                cmap="seismic",
                opacity="opacity",
                smooth_shading=True,
                ambient=0.5,
                diffuse=0.5,  # specular=0.2,
                show_scalar_bar=False,
            )
        else:
            # add a single, opaque and shiny isosurface
            pl.add_mesh(
                iso_mesh,
                opacity=1,
                scalars="quantile",
                clim=(-isosurface_quantile, isosurface_quantile),
                cmap="bwr_r",
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.5,
                show_scalar_bar=False,
            )

    if title is not None:
        pl.add_title(title)

    pl.camera_position = "iso"

    if plotter is None:  # if no plotter was passed, show the plot
        return pl.show()
    else:  # otherwise, return the plotter, e.g. to save the image next
        return pl


def plot_density(
    cube: DataCube = None,
    mode: str = "auto",
    plot_molecule: bool = True,
    isosurface_quantile: float = None,
    isosurface_opacity: float = 0.4,
    plotter: pv.Plotter = None,
    figsize: tuple[int, int] = None,
    title: str = None,
    cmap: str = None,
):
    """Plot an electron density using pyvista. By default, the orbital is plotted as a volume.

    Args:
        cube: The `DataCube`.
        mode: The mode to use for plotting. Must be one of 'volume', 'isosurface', 'nested_isosurfaces'.
        plot_molecule: Whether to plot the molecule.
        isosurface_quantile: For mode 'isosurface', the quantile of the total mass to be contained in the isosurface.
            Defaults to 0.9.
        isosurface_opacity: For mode 'isosurface', the opacity of the isosurface.
        plotter: A pyvista plotter to use. If None, a new plotter is created.
        figsize: The figure size in inches.
        title: The title of the plot.
        cmap: The colormap to use.

    Returns:
        The pyvista plotter.
    """

    if mode == "auto":
        if isosurface_quantile is None:
            mode = "volume"
        else:
            mode = "isosurface"
    assert mode in [
        "volume",
        "isosurface",
        "nested_isosurfaces",
    ], f"Invalid mode: {mode}. Must be one of 'volume', 'isosurface', 'nested_isosurfaces'."

    if plotter is not None:
        pl = plotter
    else:
        plotter_kwargs = dict(image_scale=2)
        if figsize is not None:
            dpi = int(rcParams["figure.dpi"])
            plotter_kwargs["window_size"] = [s * dpi for s in figsize]
        pl = pv.Plotter(**plotter_kwargs)

    density = cube.to_pyvista()

    if mode == "volume":
        raise NotImplementedError

    if plot_molecule:
        mol = cube.mol
        pl.add_mesh(**get_sticks_mesh_dict(mol))

    if mode in ["isosurface", "nested_isosurfaces"]:
        if isinstance(isosurface_quantile, float):
            assert (
                mode == "isosurface"
            ), f'Invalid mode: {mode}. Must be "isosurface" for a single isosurface value'
            quantiles = np.array([isosurface_quantile])
        elif mode == "nested_isosurfaces":
            n_surfaces = 20
            quantiles = np.linspace(0, 1, n_surfaces + 2)[1:-1]

        else:
            isosurface_quantile = 0.9
            quantiles = np.array([isosurface_quantile])

        isosurface_values = find_isosurface_value(density["data"], quantiles, p=1)

        isosurface = density.contour(isosurfaces=isosurface_values)
        iso_mesh = isosurface.extract_geometry()

        iso_mesh["quantile"] = np.zeros(iso_mesh.n_points)

        # iso_mesh["opacity"] = np.zeros(iso_mesh.n_points)
        # for quantile, isosurface_value in zip(quantiles, isosurface_values):
        #     mask = iso_mesh["data"] == isosurface_value
        #     iso_mesh["quantile"][mask] = quantile
        #     iso_mesh["opacity"][mask] = 1 - np.abs(quantile)

        iso_mesh["opacity"] = np.ones(iso_mesh.n_points) * 0.05

        if mode == "nested_isosurfaces":
            # add a set of nested, transparent isosurfaces
            iso_mesh["quantile"] = -np.sign(iso_mesh["quantile"]) * (
                1 - np.abs(iso_mesh["quantile"])
            )
            pl.add_mesh(
                iso_mesh,
                # scalars="quantile",
                # clim=(0, 1),
                cmap=cmap,
                opacity=0.04,  # "opacity",
                smooth_shading=True,
                ambient=0.5,
                diffuse=0.5,  # specular=0.2,
                show_scalar_bar=False,
            )
        else:
            # add a single, transparent and shiny isosurface
            pl.add_mesh(
                iso_mesh,
                opacity=isosurface_opacity,
                color="white",
                smooth_shading=True,
                ambient=0.4,
                diffuse=0.7,
                specular=0.3,
                show_scalar_bar=False,
            )

    if title is not None:
        pl.add_title(title)

    pl.camera_position = "iso"

    if plotter is None:  # if no plotter was passed, show the plot
        return pl.show()
    else:  # otherwise, return the plotter, e.g. to save the image next
        return pl

"""This module plots slices of the density of the orbitals of the kohn sham calculation and of the
density after density fitting.

Manual of the interactive plots:
 - m : change plotting mode between different metrics
 - n : toggle between comparing different metrics and different basis sets
 - b : toggle contour lines
 - Arrow keys : move the view on the x-y axis
 - x/y : move the view along the z axis
 - c/v : decrease/increase the colour range of the plots
 - k/l : zoom out/in
 - ä/ö : change the basis set on which the df options are compare. (Or vice versa)
 - w/e : rotate around x-axis
 - r/t : rotate around z-axis
 - z/u : rotate around y-axis
 - o/i : decrease/increase the resolution of the plots by a factor of 2
 - p/ü : change the orbit which is shown in the kohn_sham plot (0 shows the total density)
 - d/g : change index of atom which is being shown
 - h/j : increase/decrease the shown scf iteration
 - ,/. : show different density basis functions
 - 0,1 : Show different l basis functions
 - 8 : toggle different initial guesses to subtract from the plot
 - 9 : toggle different annotation methods to display the position of the nuclei
 - q : quit
 - s : save image
 - f : toggle fullscreen
"""


import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from pyscf import dft, gto, scf
from pyscf.dft import RKS

from mldft.datagen.methods.density_fitting import (
    get_density_fitting_function,
    ksdft_density_matrix,
)
from mldft.datagen.methods.ksdft_calculation import ksdft
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
)
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import (
    build_molecule_np,
    chem_formula_from_atomic_numbers,
    construct_aux_mol,
    load_scf,
    numbers_to_element_symbols,
    print_basis,
)


def _save_additional_data_callback(envs: dict) -> None:
    """Save the data of each iteration in the chkfile. Modified version to provide additional data
    for tests.

    Args:
        envs: Dictionary with the local environment variables of the iteration.
    Returns:
        None
    """
    cycle = envs["cycle"]
    mf = envs["mf"]
    info = {
        # Two additional variables are stored, which are useful for tests
        "fock_matrix": envs["fock"],
        "mo_energy": envs["mo_energy"],
    }
    # Load the existing data, update it with the additional data and save it again
    # Otherwise the data would be overwritten
    data = scf.chkfile.load(mf.chkfile, f"KS-iteration/{cycle:d}")
    data.update(info)
    scf.chkfile.save(mf.chkfile, f"KS-iteration/{cycle:d}", data)


class PlotDensitySlices:
    # The methods that are going to be tested
    default_mode_names = [
        "log density",
        "density",
        r"hartree $\int dr’\frac{\rho(r)\rho(r’)}{|r-r'|}$",
        r"hartree ks-of $\int dr’\frac{\rho(r)\rho(r’)}{|r-r'|}$",
        r"hartree lines $\int dr’\frac{\rho(r)\rho(r’)}{|r-r'|}$",
        r"hartree ks-of lines $\int dr’\frac{\rho(r)\rho(r’)}{|r-r'|}$",
        r"residual hartree $\int dr’\frac{(\rho(r)-\rho’(r))(\rho(r’)-\rho’(r’))}{|r-r'|}$",
        r"residual hartree lines $\int dr’\frac{(\rho(r)-\rho’(r))(\rho(r’)-\rho’(r’))}{|r-r'|}$",
        "external_energy",
        "external_energy lines",
        "external_energy ks-of",
        "external_energy ks-of lines",
        "lines_density",
        "ks-of",
        "lines ks-of",
        "(ks-of)/of",
        "curvature",
        "curvature lines",
    ]
    default_ATOM_COLORS = {
        "C": "#000000",  # c8c8c8",
        "H": "#ffffff",
        "N": "#8f8fff",
        "S": "#ffc832",
        "O": "#f00000",
        "F": "#ffff00",
        "P": "#ffa500",
        "K": "#42f4ee",
        "G": "#3f3f3f",
    }

    def __init__(
        self,
        density_fitting_methods: List[str] = None,
        density_basis_sets: List[str] = None,
        mode_names: List[str] = default_mode_names,
        ATOM_COLORS: List[str] = default_ATOM_COLORS,
    ):
        """Initialise the PlotDensitySlices class with some default parameter.

        Args:
            density_fitting_methods (list): list of methods to use for density fitting
            density_basis_sets (list): list of basis sets to use for density fitting
            mode_names (list): list of mode names to choose in visualisation
            ATOM_COLORS (list): list of atom colors to choose in visualisation
        """
        self.density_fitting_methods = density_fitting_methods
        self.basis_sets = density_basis_sets
        self.mode_names = mode_names
        self.ATOM_COLORS = ATOM_COLORS

        self.mode_changed = True  # Remembers is the last action changed the visualisation mode
        self.recalc = True  # Have the gridvalues to be recalculated?
        self.offset = np.array(
            (0, 0, 0), dtype=np.float32
        )  # The offset of the center of the visualisation grid from the center of the molecule in Bohr
        self.index = 0  # The index of df-method or basis set
        self.shift_index = 0
        self.mode = 0  # The visualisation mode that is currently used
        self.c_range = 0.01  # the range of densities that is being shown
        self.scale = 2  # the scale of the visualisation grid compared to its starting value
        self.bf_index = 0
        self.l_index = 0
        self.show_mode = 0  # 0 Shows difference of df-methods and 1 of basis sets
        self.labelmode = 0  # Controls how contour lines are being displayed
        self.n_pixel = 25  # The number of gridlines in each dimension that is displayed
        self.n_contours = 5  # The number of contour line that are being displayed
        self.rot_mat = np.eye(3, dtype=np.float32)  # Rotation that is being applied to the grid
        self.atomsize = 100  # Size of atoms in 3d-scatterplot
        self.figsize = (10, 10)  # Size of the matplotlib figure in inches
        self.dpi = 80  # dots per inch of the matplotlib figure
        self.text_size = 30  # Size of the title text
        self.sub_title_size = 30  # Size of the subtitle text above each plot
        self.annotation_size = 16
        self.n_images_x = 3  # Number of plots in the x-dimension
        self.n_images_y = 3  # Number of plots in the y-dimension
        self.dark_mode = True  # Darkmode
        self.rotate_molecule_mode = "pca"
        self.show_orbital = None
        self.show_atoms = (
            True  # if the first plot should show a scatterplot of the atoms in 3d space
        )
        self.annote_atoms = 0
        self.initial_guess_index = 0
        self.initial_guess_list = [None, "minao", "atom", "huckle", "1e", "vsap"]
        self.show_comparison_mol = True  # If the first plot should show the kohn sham density
        self.from_density_fitting_statistic = False  # If the data was loaded from a density fitting statistics class(currently not supported)
        self.comparison_mol_has_orbitals = True
        self.show_mol_list = True
        self.mol_list_has_orbitals = False
        self.colormap = "coolwarm"  # cm.jet

    def set_figure_images(self, n_images: int | tuple[int, int]):
        """Sets the number of windows to be shown.

        Args:
            n_images (int | tuple[int]): number of images to display
                if a tuple is given interpret it as (n_images_x,n_images_y)
        """
        if n_images is int:
            n_images = (n_images, 1)
        self.n_images_x, self.n_images_y = n_images

    @staticmethod
    def compute_rot_mat(mol: gto.Mole, rotate_molecule_mode: str):
        """
        Computes the rotation matrix and translation vector to rotate a molecule into the plane of most symmetry
        Args:
            mol (gto.Mole): molecule to be rotated
            rotate_molecule_mode (str): method by which the axis of most symmetry is determined
        Returns:
            offset (np.ndarray): 3d vector of translation
            rotation matrix: 3x3 rotation matrix
        """
        n_mol = len(mol.atom_charges())
        if n_mol == 1:
            return mol.atom_coords()[0], np.eye(3)
        if n_mol == 2:
            # Calculate the difference vector
            d = mol.atom_coords()[0] - mol.atom_coords()[1]
            # Calculate the angle for rotation around the y-axis
            theta_y = np.arctan2(d[2], d[0])
            R_y = np.array(
                [
                    [np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)],
                ]
            )

            # Rotate the difference vector around the y-axis
            d_rotated_y = np.dot(R_y, d)

            # Calculate the angle for rotation around the x-axis
            phi_x = np.arctan2(d_rotated_y[2], d_rotated_y[1])
            R_x = np.array(
                [[1, 0, 0], [0, np.cos(phi_x), -np.sin(phi_x)], [0, np.sin(phi_x), np.cos(phi_x)]]
            )

            # Combine the two rotations
            R = np.dot(R_x, R_y)

            return np.mean(mol.atom_coords(), axis=0), R
        if rotate_molecule_mode is None:
            offset1 = np.mean(mol.atom_coords(), axis=0)
            mat = np.eye(3)
        elif rotate_molecule_mode == "pca":
            offset1, mat = rotate_molecule_pca(mol)
        elif rotate_molecule_mode == "plane":
            offset1, mat = rotate_molecule2_onto_plane(mol)
        else:
            raise ValueError("Rotation mode must be either None, 'pca' or 'plane'")
        return offset1, mat

    @staticmethod
    def rotate_mol(mol: gto.Mole, rot_mat: np.ndarray, offset: np.ndarray) -> gto.Mole:
        """
        Given a rotation matrix and an offset, rotates and translates a molecule
        Args:
            mol (gto.Mole): molecule to be rotated
            rot_mat (np.ndarray): rotation matrix
            offset (np.ndarray): translation vector
        Returns (gto.Mole): rotated molecule
        """
        return build_molecule_np(
            mol.atom_charges(),
            (mol.atom_coords() - offset) @ rot_mat,
            basis=mol.basis,
            unit="Bohr",
        )

    def get_initial_guess_density(self, grid_points, mol, key):
        gamma_0 = RKS(mol).get_init_guess(mol, key)
        density_0 = self.calcualte_density_from_orbitals(grid_points, mol, gamma_0)
        return density_0

    @classmethod
    def show_mol_atoms(cls, mol: gto.Mole, rotate_molecule_mode: str = "pca"):
        """
        Plots the positions of the atoms of a molecule in 3d space
        Args:
             mol (gto.Mole): molecule to be shown
             rotate_molecule_mode (str, optional): 'pca' or 'rotated' or None. Defaults to 'pca'.
        Returns:
            PlotDensitySlices object adjusted to show the given molecule
        """
        pds = cls(
            density_basis_sets=["None"],
            density_fitting_methods=["None"],
            mode_names=["Default"],
        )
        pds.set_figure_images((1, 1))
        pds.rotate_molecule_mode = rotate_molecule_mode
        pds.show_atoms = True
        pds.show_comparison_mol = False
        offset1, mat = pds.compute_rot_mat(mol, rotate_molecule_mode)
        pds.comparison_mol = pds.rotate_mol(mol, mat, offset1)
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = pds.comparison_mol.atom_coords()
        pds.scf_iter = 0
        pds.mol_number = 0
        pds.mol_with_density_basis = []
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        return pds

    @classmethod
    def show_mol_ks_density(
        cls,
        mol: gto.Mole,
        gamma: np.ndarray = None,
        rotate_molecule_mode: str = "pca",
        iteration: int = -1,
    ):
        """
        Creates plots of slices of the density of a molecules gives as a gto.Mole and orbital coefficients
        Args:
            mol (gto.Mole): The molecule which is supposed to been shown
            gamma (np.ndarray, Optional): The orbital coefficients of the molecule from the kohn sham calculation(if None they are calculated on the spot)
            rotate_molecule_mode (str, optional): Whether to rotate the molecules according to the rotation mode
            iteration (int, optional): The index of the kohn sham step to be shown

        Returns:
            PlotDensitySlices object adjusted to show the given molecule
        """
        offset1, mat = PlotDensitySlices.compute_rot_mat(mol, rotate_molecule_mode)
        mol = PlotDensitySlices.rotate_mol(mol, mat, offset1)
        if gamma is None:
            logger.info(
                "Gamma matrix not found, starting kohn sham calculation to compute ground state density."
            )
            with TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                filesave_path = temp_dir_path / "test.chk"
                assert not filesave_path.exists()
                ksdft(
                    mol,
                    filesave_path,
                    diis_method="CDIIS",
                    extra_callback=_save_additional_data_callback,
                )

                assert filesave_path.exists()
                results, initialisation, data_of_iteration = load_scf(filesave_path)
                mo_coeff = data_of_iteration[iteration]["molecular_coeffs_orbitals"]
                mo_occ = data_of_iteration[iteration]["occupation_numbers_orbitals"]
                gamma = ksdft_density_matrix(mo_coeff, mo_occ)

        pds = cls(
            density_basis_sets=["None"],
            density_fitting_methods=["None"],
            mode_names=[
                "log density",
                "density",
                "lines_density",
                "curvature",
                r"hartree $\int dr’\frac{\rho(r)\rho(r’)}{|r-r'|}$",
            ],
        )
        pds.set_figure_images((1, 2))
        pds.rotate_molecule_mode = rotate_molecule_mode
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = True
        pds.comparison_mol = mol
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = pds.comparison_mol.atom_coords()
        pds.gamma = gamma
        pds.atomic_numbers = np.asarray(mol.atom_charges())
        pds.atom_pos = mol.atom_coords()
        pds.scf_iter = -1
        pds.mol_number = 0
        pds.mol_with_density_basis = []
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        return pds

    @classmethod
    def show_mol_ks_and_of_density(
        cls,
        mol: gto.Mole,
        gamma: np.ndarray = None,
        rotate_molecule_mode: str = "pca",
        iteration: int = -1,
        basis_sets: List[str] = ("even_tempered_2.5",),
        density_fitting_methods: List[str] = ("hartree+external_mofdft",),
        of_coeffs_list: List[List[np.ndarray]] = None,
        mol_with_density_basis: List[gto.Mole] = None,
    ):
        """
        Creates plots of slices of the density of a molecule given as a gto.Mole and orbital coefficients as well
        as an orbital free basis and coefficients
        Args:
            mol (gto.Mole): The molecule which is supposed to been shown
            gamma (np.ndarray, Optional): The orbital coefficients of the molecule from the kohn sham calculation(if None they are calculated on the spot)
            rotate_molecule_mode (str, optional): Whether to rotate the molecules according to the rotation mode
            iteration (int, optional): The index of the kohn sham step to be shown
            basis_sets (List[str], optional): Names of basis sets which should be used to fit the orbital density
            density_fitting_methods (List[str], optional): Names of fitting methods which should be used to fit the density
            of_coeffs_list (List[List[np.ndarray]]): List of the orbital free coefficients of the molecules(if None they are calculated on the spot)
            mol_with_density_basis(List[gto.Mole]): The molecule using of basis sets

        Returns:
            PlotDensitySlices object adjusted to show the given molecule
        """
        offset1, mat = PlotDensitySlices.compute_rot_mat(mol, rotate_molecule_mode)
        mol = PlotDensitySlices.rotate_mol(mol, mat, offset1)
        if gamma is None:
            logger.info(
                "Gamma matrix not found, starting kohn sham calculation to compute ground state density."
            )
            with TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                filesave_path = temp_dir_path / "test.chk"
                assert not filesave_path.exists()
                ksdft(
                    mol,
                    filesave_path,
                    diis_method="CDIIS",
                    extra_callback=_save_additional_data_callback,
                )

                assert filesave_path.exists()
                results, initialisation, data_of_iteration = load_scf(filesave_path)
                mo_coeff = data_of_iteration[iteration]["molecular_coeffs_orbitals"]
                mo_occ = data_of_iteration[iteration]["occupation_numbers_orbitals"]
                gamma = ksdft_density_matrix(mo_coeff, mo_occ)
        if mol_with_density_basis is None or of_coeffs_list is None:
            mol_with_density_basis = []
            temp_of_coeffs_list = []
            for i, name in enumerate(basis_sets):
                of_coeffs = []

                # Build molecules for each basis set
                aux_mol = construct_aux_mol(mol, aux_basis_name=name)
                aux_mol.build()
                logger.info(
                    f"Basis set: {name} with n_ao = {aux_mol.nao} and the following Orbitals:"
                )
                print_basis(aux_mol)
                # df.addons.make_auxmol(mol_with_orbital_basis, auxbasis=name)
                # build list of of-coeffs for all df-methods for this basis set
                mol_with_density_basis.append(aux_mol)
                if of_coeffs_list is None:
                    for j, method in enumerate(density_fitting_methods):
                        W_coulomb = np.asarray(get_coulomb_matrix(aux_mol))
                        L_coulomb = get_coulomb_tensor(aux_mol, mol)
                        external_potential_p = np.asarray(get_nuclear_attraction_vector(aux_mol))
                        external_potential_c = np.asarray(get_nuclear_attraction_matrix(mol))

                        of_coeffs.append(
                            get_density_fitting_function(
                                method,
                                mol,
                                aux_mol,
                                W_coulomb,
                                L_coulomb,
                                external_potential_c,
                                external_potential_p,
                            )(gamma)
                        )
                    temp_of_coeffs_list.append(of_coeffs)
        else:
            mol_with_density_basis = [
                PlotDensitySlices.rotate_mol(mol, mat, offset1) for mol in mol_with_density_basis
            ]
        if of_coeffs_list is None:
            of_coeffs_list = temp_of_coeffs_list
        pds = cls(
            density_basis_sets=basis_sets,
            density_fitting_methods=density_fitting_methods,
            mode_names=[
                "log density",
                "density",
                "lines_density",
                "ks-of",
                "lines ks-of",
                "(ks-of)/of",
            ],
        )
        pds.set_figure_images((1, 3))
        pds.rotate_molecule_mode = rotate_molecule_mode
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = True
        pds.mol_list_has_orbitals = False
        pds.comparison_mol = mol
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = pds.comparison_mol.atom_coords()
        pds.gamma = gamma
        pds.atomic_numbers = np.asarray(mol.atom_charges())
        pds.atom_pos = mol.atom_coords()
        pds.scf_iter = -1
        pds.mol_number = 0
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        pds.of_coeffs_list = of_coeffs_list
        pds.mol_with_density_basis = mol_with_density_basis
        return pds

    @classmethod
    def show_mol_of_density(
        cls,
        of_coeffs: np.ndarray,
        mol_with_density_basis: gto.Mole,
        rotate_molecule_mode="pca",
        basis_set: str = "even_tempered_2.5",
    ):
        """
        Creates plots of slices of the density of a molecules gives as a gto.Mole object and coefficients
        Args:
            of_coeffs (np.ndarray): The orbital free coefficients of the molecule
            mol_with_density_basis (gto.Mole): The molecule which is supposed to been shown
            rotate_molecule_mode (str, optional): Whether to rotate the molecules according to the rotation mode
            basis_set (str, optional): Name of basis set in which the of-data-object was constructed
        Returns:
            PlotDensitySlices object adjusted to show the given molecule
        """
        offset1, mat = PlotDensitySlices.compute_rot_mat(
            mol_with_density_basis, rotate_molecule_mode
        )
        mol = PlotDensitySlices.rotate_mol(mol_with_density_basis, mat, offset1)
        pds = cls(
            density_basis_sets=[basis_set],
            density_fitting_methods=[""],
            mode_names=["log density", "density", "lines_density"],
        )
        pds.set_figure_images((1, 2))
        pds.rotate_molecule_mode = "None"
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = False
        pds.show_mol_list = False
        pds.comparison_mol = mol
        pds.of_coeffs = of_coeffs
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = mol.atom_coords()
        pds.atomic_numbers = np.asarray(mol.atom_charges())
        pds.atom_pos = mol.atom_coords()
        pds.scf_iter = -1
        pds.mol_number = 0
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        pds.of_coeffs_list = [[]]
        pds.mol_with_density_basis = []
        return pds

    @classmethod
    def show_of_data(
        cls,
        of_data: OFData,
        construct_ks_data=False,
        rotate_molecule_mode="pca",
        basis_set_name="even_tempered_2.5",
    ):
        """
        Creates plots of slices of the density of a molecules gives as a OFData object
        Args:
            of_data (OFData): OFData object containing the molecule
            construct_ks_data (bool, optional): Whether to redo the ks-calculations to compare the of density against
            rotate_molecule_mode (str, optional): Whether to rotate the molecules according to the rotation mode
            basis_set_name (str, optional): Name of basis set in which the of-data-object was constructed
        Returns:
            PlotDensitySlices object adjusted to show the given molecule
        """
        of_coeffs = of_data.coeffs
        mol_with_orbital_basis = build_molecule_np(
            of_data.atomic_numbers, np.asarray(of_data.pos), of_data.ks_basis, unit="Bohr"
        )
        mol_with_density_basis = construct_aux_mol(mol_with_orbital_basis, basis_set_name)
        if not construct_ks_data:
            return cls.show_mol_of_density(
                of_coeffs, mol_with_density_basis, rotate_molecule_mode, basis_set_name
            )
        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            filesave_path = temp_dir_path / "test.chk"
            assert not filesave_path.exists()
            ksdft(
                mol_with_orbital_basis,
                filesave_path,
                diis_method="CDIIS",
                extra_callback=_save_additional_data_callback,
            )

            assert filesave_path.exists()
            results, initialisation, data_of_iteration = load_scf(filesave_path)
            mo_coeff = data_of_iteration[of_data.scf_iteration]["molecular_coeffs_orbitals"]
            mo_occ = data_of_iteration[of_data.scf_iteration]["occupation_numbers_orbitals"]
            gamma = ksdft_density_matrix(mo_coeff, mo_occ)
        pds = cls.show_mol_ks_and_of_density(
            mol_with_orbital_basis,
            gamma,
            rotate_molecule_mode=rotate_molecule_mode,
            iteration=of_data.scf_iteration,
            basis_sets=[
                basis_set_name,
            ],
            density_fitting_methods=("",),
            of_coeffs_list=[[of_coeffs]],
            mol_with_density_basis=[mol_with_density_basis],
        )
        return pds

    @classmethod
    def show_difference_orbital_densities(
        cls,
        ground_truth_mol: gto.Mole,
        gound_truth_gamma: np.ndarray,
        mol_list: List[gto.Mole],
        gamma_list: List[np.ndarray],
        names_basis_sets: List[str],
        rotate_molecule_mode="pca",
    ):
        assert (
            len(mol_list) == len(gamma_list) == len(names_basis_sets)
        ), f"mol_list and gamma_list should be the same length but got length ({len(mol_list)}) and ({len(gamma_list)})"
        offset1, mat = PlotDensitySlices.compute_rot_mat(ground_truth_mol, rotate_molecule_mode)
        ground_truth_mol = PlotDensitySlices.rotate_mol(ground_truth_mol, mat, offset1)
        mol_list = [PlotDensitySlices.rotate_mol(mol, mat, offset1) for mol in mol_list]
        pds = cls(
            density_basis_sets=names_basis_sets,
            density_fitting_methods=[""],
            mode_names=[
                "log density",
                "density",
                "lines_density",
                "ks-of",
                "lines ks-of",
                "(ks-of)/of",
            ],
        )
        pds.set_figure_images((1, 2 + len(mol_list)))
        pds.rotate_molecule_mode = "None"
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = True
        pds.mol_list_has_orbitals = True
        pds.show_mol_list = True
        pds.comparison_mol = ground_truth_mol
        pds.gamma = gound_truth_gamma
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.atomic_numbers = np.asarray(ground_truth_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.scf_iter = -1
        pds.mol_number = 0
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        pds.of_coeffs_list = [gamma_list]
        pds.mol_with_density_basis = mol_list
        pds.comparison_mols_with_orbital_basis = mol_list
        pds.comparison_mols_gamma_list = gamma_list
        pds.compare_kohn_sham_orbitals = True
        pds.show_mode = 1
        return pds

    @classmethod
    def show_difference_of_densities(
        cls,
        ground_truth_mol: gto.Mole,
        gound_truth_densities: np.ndarray,
        mol_list: List[gto.Mole],
        densities_list: List[np.ndarray],
        names_basis_sets: List[str],
        rotate_molecule_mode="pca",
    ):
        assert (
            len(mol_list) == len(densities_list) == len(names_basis_sets)
        ), f"mol_list and gamma_list should be the same length but got length ({len(mol_list)}) and ({len(densities_list)})"
        offset1, mat = PlotDensitySlices.compute_rot_mat(ground_truth_mol, rotate_molecule_mode)
        ground_truth_mol = PlotDensitySlices.rotate_mol(ground_truth_mol, mat, offset1)
        mol_list = [PlotDensitySlices.rotate_mol(mol, mat, offset1) for mol in mol_list]
        pds = cls(
            density_basis_sets=names_basis_sets,
            density_fitting_methods=[""],
            mode_names=[
                "log density",
                "density",
                "lines_density",
                "ks-of",
                "lines ks-of",
                "(ks-of)/of",
            ],
        )
        pds.set_figure_images((1, 2 + len(mol_list)))
        pds.rotate_molecule_mode = "None"
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = False
        pds.comparison_mol = ground_truth_mol
        pds.of_coeffs = gound_truth_densities
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.atomic_numbers = np.asarray(ground_truth_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.scf_iter = -1
        pds.mol_number = 0
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        pds.of_coeffs_list = [[d] for d in densities_list]
        pds.mol_with_density_basis = mol_list
        pds.comparison_mols_with_orbital_basis = mol_list
        pds.compare_kohn_sham_orbitals = False
        pds.show_mode = 1
        return pds

    @classmethod
    def show_difference_of_densities_same_basis(
        cls,
        ground_truth_mol: gto.Mole,
        gound_truth_densities: np.ndarray,
        densities_list: List[np.ndarray],
        names_basis_sets: List[str],
        rotate_molecule_mode="pca",
    ):
        assert len(densities_list) == len(
            names_basis_sets
        ), f"mol_list and names_basis_sets should be the same length but got length ({len(names_basis_sets)}) and ({len(densities_list)})"
        offset1, mat = PlotDensitySlices.compute_rot_mat(ground_truth_mol, rotate_molecule_mode)
        ground_truth_mol = PlotDensitySlices.rotate_mol(ground_truth_mol, mat, offset1)
        pds = cls(
            density_basis_sets=names_basis_sets,
            density_fitting_methods=[""],
            mode_names=[
                "log density",
                "density",
                "lines_density",
                "ks-of",
                "lines ks-of",
                "(ks-of)/of",
            ],
        )
        pds.set_figure_images((1, 2 + len(densities_list)))
        pds.rotate_molecule_mode = "None"
        pds.show_atoms = True
        pds.show_comparison_mol = True
        pds.comparison_mol_has_orbitals = False
        pds.comparison_mol = ground_truth_mol
        pds.of_coeffs = gound_truth_densities
        pds.atomic_numbers = np.asarray(pds.comparison_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.atomic_numbers = np.asarray(ground_truth_mol.atom_charges())
        pds.atom_pos = ground_truth_mol.atom_coords()
        pds.scf_iter = 0
        pds.mol_number = 0
        pds.mol_name = chem_formula_from_atomic_numbers(pds.atomic_numbers)
        pds.of_coeffs_list = [densities_list]
        pds.mol_with_density_basis = [ground_truth_mol]
        pds.comparison_mols_with_orbital_basis = [ground_truth_mol]
        pds.compare_kohn_sham_orbitals = False
        pds.show_mode = 0
        return pds

    def plot_contour(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        title: str,
        borders: List[float],
        show_full_scale: bool = False,
    ):
        """Plot a contour or line plot into a figure.

        Args:
            fig (Figure): Figure instance
            ax (Axes): Axes instance
            x (ndarray): x coordinates
            y (ndarray): y coordinates
            Z (ndarray): density values at (x,y) shape = (x.shape[0],y.shape[0])
            title (str): Title of the plot
            borders (List[float]): Borders of the plot
            show_full_scale (bool): set the range such the whole value range is in view
        """
        ax.set_title(title, fontsize=self.sub_title_size)
        if self.mode_changed:
            self.c_range = np.max(np.abs(Z))
        if show_full_scale:
            local_c_range = np.max(np.abs(Z))
        else:
            local_c_range = self.c_range

        max = np.max(np.abs(Z))
        min = np.min(Z)
        max = np.max(np.array(max, -min))
        if "lines" in self.current_mode:
            norm = mpl.colors.Normalize(vmin=y.min(), vmax=y.max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
            cmap.set_array([])
            for i in range(Z.shape[0]):
                ax.plot(x, Z[i, :], c=cmap.to_rgba(y[i]))
                if (min > 0) and self.bf_index == 0:  # "density" in self.current_mode or
                    ax.set_ylim((0, local_c_range))
                else:
                    ax.set_ylim((-local_c_range, local_c_range))
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.001,
                    ax.get_position().y0,
                    0.005,
                    ax.get_position().height,
                ]
            )
            fig.colorbar(cmap, ticks=y, ax=ax, cax=cax)
        elif (min > 0) and self.bf_index == 0:  # "density" in self.current_mode
            levels = np.linspace(np.log(min), np.log(max), num=self.n_contours)
            levels = np.exp(levels)
            if "log" in self.current_mode:
                im = ax.imshow(
                    np.log(Z * (Z > 0)),
                    interpolation="bilinear",
                    origin="lower",
                    cmap=self.colormap,
                    extent=borders,
                )
            else:
                im = ax.imshow(
                    Z,
                    interpolation="bilinear",
                    origin="lower",
                    cmap=self.colormap,
                    extent=borders,
                    vmax=local_c_range,
                    vmin=0,
                )
        else:
            levels = np.linspace(-local_c_range, local_c_range, num=self.n_contours)
            # mean = np.sqrt(np.mean(Z**2))
            # Z = np.clip(Z,-mean,mean)
            im = ax.imshow(
                Z,
                interpolation="bilinear",
                origin="lower",
                cmap=self.colormap,
                extent=borders,
                vmin=-local_c_range,
                vmax=local_c_range,
            )
        ax.set_xlabel("x in Bohr")
        ax.set_ylabel("y in Bohr")
        if "lines" not in self.current_mode and self.labelmode > 0:
            CS = ax.contour(
                x,
                y,
                Z,
                levels,
                origin="lower",
                cmap=self.colormap,
                extend="both",
            )

            # Thicken the zero contour.
            lws = np.resize(CS.get_linewidth(), len(levels))
            CS.set_linewidth(lws)
            if self.labelmode == 2:
                ax.clabel(
                    CS, levels, inline=True, fmt="%1.1e", fontsize=14
                )  # label every second level

            # make a colorbar for the contour lines
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.02,
                    ax.get_position().y0,
                    0.005,
                    ax.get_position().height,
                ]
            )
            CB = fig.colorbar(CS, orientation="vertical", cax=cax)
            CB.ax.set_yticklabels([f"{i:1.1e}" for i in levels])
            # l, b, w, h = ax.get_position().bounds
            # ll, bb, ww, hh = CB.ax.get_position().bounds
            # CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])

        # ax.set_title('Lines with colorbar')
        if "lines" not in self.current_mode:
            if self.annote_atoms > 0:
                atom_on_plot = np.logical_and(
                    np.abs(self.atom_pos_in_local_coord_system[:, 0]) < self.max_range,
                    np.abs(self.atom_pos_in_local_coord_system[:, 1]) < self.max_range,
                )
                if atom_on_plot.any():
                    atom_pos_in_local_coord_system = self.atom_pos_in_local_coord_system[
                        atom_on_plot
                    ]
                    alpha = 1 - np.abs(atom_pos_in_local_coord_system[:, 2]) / self.max_range
                    alpha = alpha * (alpha > 0)
                    color = self.atomic_colors[atom_on_plot]
                    if self.annote_atoms >= 3:
                        alpha = 1.0
                        color = "black"
                    ax.scatter(
                        atom_pos_in_local_coord_system[:, 0],
                        atom_pos_in_local_coord_system[:, 1],
                        marker="+",
                        c=color,
                        alpha=alpha,
                    )
                    if self.annote_atoms == 2:
                        for i, (name, pos) in enumerate(
                            zip(
                                np.array(self.atom_names)[atom_on_plot],
                                atom_pos_in_local_coord_system,
                            )
                        ):
                            ax.text(
                                x=pos[0],
                                y=pos[1],
                                s=" " + name,
                                alpha=alpha[i],
                                color=color[i],
                                size=self.annotation_size,
                                path_effects=[pe.withStroke(linewidth=1, foreground="black")],
                            )
                    elif self.annote_atoms == 3:
                        for name, pos in zip(
                            np.array(self.atom_names)[atom_on_plot], atom_pos_in_local_coord_system
                        ):
                            ax.text(
                                x=pos[0],
                                y=pos[1],
                                s=name + f" {pos[2]:2e}",
                                color="black",
                                size=self.annotation_size,
                            )
                    elif self.annote_atoms == 4:
                        for name, pos in zip(
                            np.array(self.atom_names)[atom_on_plot], atom_pos_in_local_coord_system
                        ):
                            ax.text(
                                x=pos[0],
                                y=pos[1],
                                s=" " + name,
                                color="black",
                                size=self.annotation_size,
                            )
            # We can still add a colorbar for the image, too.
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.001,
                    ax.get_position().y0,
                    0.005,
                    ax.get_position().height,
                ]
            )
            CBI = fig.colorbar(im, orientation="vertical", cax=cax)
            CBI.set_label(r"Density in electrons/Bohr$^{3}$")

    def calcualte_density_from_orbitals(
        self, gridpoints: np.ndarray, mol_with_orbital_basis: gto.Mole, gamma: np.ndarray
    ):
        n_ao = mol_with_orbital_basis.nao
        if self.current_mode == "curvature":
            # mol_ks.kart = False
            ao_ks_derivs = dft.numint.eval_ao(
                mol_with_orbital_basis,
                gridpoints.reshape(3, self.n_pixel * self.n_pixel).T,
                deriv=2,
            )
            ao_ks_derivs = np.transpose(ao_ks_derivs, (0, 2, 1)).reshape(
                10, n_ao, self.n_pixel, self.n_pixel
            )
            ao_ks = ao_ks_derivs[0]
            # 0,x,y,z,xx,xy,xz,yy,yz,zz
            ao_ks_xx_yy_zz = ao_ks_derivs[4] + ao_ks_derivs[7] + ao_ks_derivs[9]
            ao_ks_x, ao_ks_y, ao_ks_z = ao_ks_derivs[1], ao_ks_derivs[2], ao_ks_derivs[3]
            return (
                2 * np.einsum("ijk,il,ljk->jk", ao_ks_xx_yy_zz, gamma, ao_ks, optimize=True)
                + 2 * np.einsum("ijk,il,ljk->jk", ao_ks_x, gamma, ao_ks_x, optimize=True)
                + 2 * np.einsum("ijk,il,ljk->jk", ao_ks_y, gamma, ao_ks_y, optimize=True)
                + 2 * np.einsum("ijk,il,ljk->jk", ao_ks_z, gamma, ao_ks_z, optimize=True)
            )
        elif "external" in self.current_mode:
            ao_ks = dft.numint.eval_ao(
                mol_with_orbital_basis,
                gridpoints.reshape(3, self.n_pixel * self.n_pixel).T,
            ).T.reshape(n_ao, self.n_pixel, self.n_pixel)
            return np.einsum("ijk,il,ljk->jk", ao_ks, gamma, ao_ks, optimize=True)
            external_potential = np.einsum(
                "ijk,il,ljk->jk",
                ao_ks,
                get_nuclear_attraction_matrix(mol_with_orbital_basis),
                ao_ks,
                optimize=True,
            )
            return external_potential * self.density_ks
        elif "hartree" in self.current_mode:
            grid_orbitals = grid_setup(mol_with_orbital_basis, grid_level=1)
            ao_orbitals = dft.numint.eval_ao(mol_with_orbital_basis, grid_orbitals.coords, deriv=0)
            if not hasattr(self, "rho_ks_on_grid"):
                self.rho_ks_on_grid = np.einsum(
                    "ij,jk,ik->i", ao_orbitals, gamma, ao_orbitals, optimize=True
                )

            coords = grid_orbitals.coords
            gridpoints_shaped = gridpoints.reshape(3, self.n_pixel * self.n_pixel).T
            distances = (
                np.sqrt(np.sum((coords[None, :, :] - gridpoints_shaped[:, None, :]) ** 2, axis=2))
                + 1e-12
            )
            # nonzero_mask = distances != 0
            # torch.div(1, distances[nonzero_mask], out=distances[nonzero_mask])
            # self.potential = np.asarray(distances)
            self.potential = 1 / distances
            # self.potential[distances == 0] = 0.

            integrated_density = np.einsum(
                "ij,j,j->i",
                self.potential,
                self.rho_ks_on_grid,
                grid_orbitals.weights,
                optimize=True,
            )

            ao_ks = dft.numint.eval_ao(
                mol_with_orbital_basis,
                gridpoints.reshape(3, self.n_pixel * self.n_pixel).T,
            ).T.reshape(n_ao, self.n_pixel, self.n_pixel)
            self.density_ks_on_points = np.einsum(
                "ijk,il,ljk->jk", ao_ks, gamma, ao_ks, optimize=True
            )
            return (
                integrated_density.reshape(self.n_pixel, self.n_pixel) * self.density_ks_on_points
            )
        else:
            ao_ks = dft.numint.eval_ao(
                mol_with_orbital_basis,
                gridpoints.reshape(3, self.n_pixel * self.n_pixel).T,
            ).T.reshape(n_ao, self.n_pixel, self.n_pixel)
            return np.einsum("ijk,il,ljk->jk", ao_ks, gamma, ao_ks, optimize=True)

    def plot_kohn_sham(
        self,
        fig,
        ax,
        x_: np.ndarray,
        y_: np.ndarray,
        gridpoints: np.ndarray,
        mol_with_orbital_basis: gto.Mole,
        gamma: np.ndarray,
        show_full_range: bool,
    ):
        if self.recalc:
            self.density_ks = self.calcualte_density_from_orbitals(
                gridpoints, mol_with_orbital_basis, gamma
            )
        if self.initial_guess_list[self.initial_guess_index] is not None:
            viewer_density = self.density_ks - self.initial_guess_density
        else:
            viewer_density = self.density_ks
        # ax3.contour(x_, y_, np.log(density_ks),levels=100)

        self.plot_contour(
            fig,
            ax,
            x_,
            y_,
            viewer_density,
            "Density of the KS",
            [-self.max_range, +self.max_range, -self.max_range, +self.max_range],
            show_full_scale=show_full_range,
        )

    def plot_orbital_free_density(
        self,
        fig,
        ax,
        x_: np.ndarray,
        y_: np.ndarray,
        gridpoints: np.ndarray,
        mol_with_density_basis: gto.Mole,
        of_coeffs: np.ndarray,
        show_full_range: bool,
    ):
        if self.recalc:
            ao_p = dft.numint.eval_ao(
                mol_with_density_basis,
                gridpoints.reshape(3, self.n_pixel * self.n_pixel).T,
            ).T.reshape(mol_with_density_basis.nao, self.n_pixel, self.n_pixel)
            # gridpoints.reshape(3, self.n_pixel * self.n_pixel).T
            # )
            self.density_ks = self.calculate_of_density(mol_with_density_basis, of_coeffs, ao_p)
        if self.initial_guess_list[self.initial_guess_index] is not None:
            viewer_density = self.density_ks - self.initial_guess_density
        else:
            viewer_density = self.density_ks
        # ax3.contour(x_, y_, np.log(density_ks),levels=100)

        self.plot_contour(
            fig,
            ax,
            x_,
            y_,
            viewer_density,
            "Orbital free density",
            [-self.max_range, +self.max_range, -self.max_range, +self.max_range],
            show_full_scale=show_full_range,
        )

    def calculate_of_density(self, mol_with_density_basis, of_coeffs, ao_p):
        if self.recalc:
            basis_nao = mol_with_density_basis.nao
            if self.bf_index == 0:
                if self.l_index != 0:
                    basis_info = BasisInfo.from_mol(mol_with_density_basis)
                    l_tot = np.concatenate(
                        [
                            basis_info.l_per_basis_func[
                                basis_info.atom_ind_to_basis_function_ind[
                                    basis_info.atomic_number_to_atom_index[atomic_number]
                                ]
                            ]
                            for atomic_number in self.comparison_mol.atom_charges()
                        ]
                    )
                if self.l_index == 0:
                    basis_function_mask = np.ones(basis_nao, dtype=np.float32)
                elif self.l_index > 0:
                    basis_function_mask = l_tot == self.l_index - 1
                elif self.l_index < 0:
                    basis_function_mask = l_tot != -self.l_index - 1

            elif self.bf_index > 0:
                bf_index = (self.bf_index - 1) % basis_nao
                basis_function_mask = np.zeros(basis_nao, dtype=np.float32)
                basis_function_mask[bf_index] = 1.0
            else:
                bf_index = (-self.bf_index - 1) % basis_nao
                basis_function_mask = np.zeros(basis_nao, dtype=np.float32)
                basis_function_mask[bf_index] = np.max(of_coeffs) / of_coeffs[bf_index]
            if "residual hartree" in self.current_mode:
                grid = grid_setup(mol_with_density_basis, grid_level=1)
                ao_densities = dft.numint.eval_ao(mol_with_density_basis, grid.coords, deriv=0)
                rho_of_on_grid = np.einsum(
                    "ij,j->i",
                    ao_densities,
                    of_coeffs,
                    optimize=True,
                )
                integrated_density = np.einsum(
                    "ij,j,j->i",
                    self.potential,
                    self.rho_ks_on_grid - rho_of_on_grid,
                    grid.weights,
                    optimize=True,
                )
                return integrated_density.reshape(self.n_pixel, self.n_pixel) * (
                    self.density_ks_on_points
                    - np.einsum(
                        "i,ijk->jk",
                        of_coeffs,
                        ao_p,
                        optimize=True,
                    )
                )
            elif "hartree" in self.current_mode:
                grid = grid_setup(mol_with_density_basis, grid_level=1)
                ao_densities = dft.numint.eval_ao(mol_with_density_basis, grid.coords, deriv=0)
                rho_of_on_grid = np.einsum(
                    "ij,j->i",
                    ao_densities,
                    of_coeffs,
                    optimize=True,
                )
                integrated_density = np.einsum(
                    "ij,j,j->i",
                    self.potential,
                    rho_of_on_grid,
                    grid.weights,
                    optimize=True,
                )
                return integrated_density.reshape(self.n_pixel, self.n_pixel) * np.einsum(
                    "i,ijk->jk",
                    of_coeffs,
                    ao_p,
                    optimize=True,
                )
            elif "external" in self.current_mode:
                temp = np.einsum(
                    "i,ijk->jk",
                    of_coeffs,
                    ao_p,
                    optimize=True,
                )
                return (
                    np.einsum(
                        "i,ijk->jk",
                        get_nuclear_attraction_vector(mol_with_density_basis),
                        ao_p,
                        optimize=True,
                    )
                    * basis_function_mask
                    * temp
                )
            else:
                return np.einsum(
                    "i,ijk->jk",
                    basis_function_mask * of_coeffs,
                    ao_p,
                    optimize=True,
                )

    def plot_fig(self):
        """Fill a figure with the visualisation plots."""
        self.current_mode = self.mode_names[self.mode]
        if self.show_orbital is None:
            orbital_info = ""
        else:
            orbital_info = (
                f"KS Orbital: {self.show_orbital+1}/{self.largest_occ_orbit}({self.n_orbitals})"
            )
        l_dict = ("s", "p", "d", "f", "g", "h", "i", "j", "k")
        if self.l_index == 0:
            l_info = ""
        elif self.l_index > 0:
            l_info = f" Showing only {l_dict[self.l_index-1]} basis functions"
        else:
            l_info = f" Masking {l_dict[-self.l_index-1]} basis functions"
        inital_guess_info = ""
        if self.initial_guess_list[self.initial_guess_index] is not None:
            inital_guess_info = (
                str(self.initial_guess_list[self.initial_guess_index]) + "guess subtracted"
            )
        title = f"Density slice through {self.mol_name} (offset={self.offset}, scale={self.scale})  {self.current_mode} Iteration:{self.scf_iter} Molecule:{self.mol_number} {orbital_info} {l_info} {inital_guess_info}"
        self.fig.suptitle(title, fontsize=self.text_size)
        fig_number = self.n_images_x * 100 + self.n_images_y * 10 + self.show_atoms - 1
        max_images = self.n_images_x * self.n_images_y
        if self.show_mode == 0:
            self.n_images = min(
                len(self.density_fitting_methods),
                max_images - self.show_comparison_mol - self.show_atoms,
            )
            basis_index = np.ones(self.n_images, dtype=np.int32) * (
                self.index % len(self.basis_sets)
            )
            method_index = (np.arange(self.n_images, dtype=np.int32) + self.shift_index) % len(
                self.density_fitting_methods
            )
        elif self.show_mode == 1:
            self.n_images = min(
                len(self.basis_sets), max_images - self.show_comparison_mol - self.show_atoms
            )
            method_index = np.ones(self.n_images, dtype=np.int32) * (
                self.index % len(self.density_fitting_methods)
            )
            basis_index = (np.arange(self.n_images, dtype=np.int32) + self.shift_index) % len(
                self.basis_sets
            )

        # get the difference between all atom positions
        diff = self.atom_pos[:, :, None] - self.atom_pos[:, None, :]
        # get the maximum distance between atoms
        dist = np.sqrt(np.sum(diff**2, axis=0))
        max_range = dist.max().item() / 2

        colors = np.asarray(
            [
                self.ATOM_COLORS[numbers_to_element_symbols[i + 1]]
                if numbers_to_element_symbols[i + 1] in self.ATOM_COLORS
                else "#000000"
                for i in range(self.atomic_numbers.max())
            ]
        )
        self.max_range = max_range * self.scale

        self.atomic_colors = colors[self.atomic_numbers - 1]
        # mol_on_density = build_mol_with_even_tempered_basis(mol)
        # Make a 2d grid of 3d points in the xy-plane
        x_ = np.linspace(-self.max_range, self.max_range, self.n_pixel)
        y_ = np.linspace(-self.max_range, self.max_range, self.n_pixel)
        z = np.zeros((self.n_pixel, self.n_pixel))
        x, y = np.meshgrid(x_, y_)
        gridpoints = (
            np.einsum("ij,jlm->ilm", self.rot_mat, np.stack((x, y, z)))
            + self.offset[:, None, None]
        )
        if self.show_atoms:
            ax1 = self.fig.add_subplot(fig_number + 1, projection="3d")
            ax1.set_xlim(self.offset[0] - self.max_range, self.offset[0] + self.max_range)
            ax1.set_ylim(self.offset[1] - self.max_range, self.offset[1] + self.max_range)
            ax1.set_zlim(self.offset[2] - self.max_range, self.offset[2] + self.max_range)
            ax1.scatter(
                self.atom_pos[:, 0],
                self.atom_pos[:, 1],
                self.atom_pos[:, 2],
                s=self.atomsize,
                c=self.atomic_colors,
            )
            ax1.plot_surface(gridpoints[0], gridpoints[1], gridpoints[2], color="white", alpha=0.7)
            ax1.set_xlabel("x in Bohr")
            ax1.set_ylabel("y in Bohr")
            ax1.set_zlabel("z in Bohr")

        self.atom_names = [numbers_to_element_symbols[charge] for charge in self.atomic_numbers]
        self.atom_pos_in_local_coord_system = (
            np.einsum("ji,lj->li", self.rot_mat, self.atom_pos)
        ) - self.offset[
            None, :
        ]  # /self.max_range

        show_full_range = ("ks-of" in self.current_mode) or ("(ks-of)/of" in self.current_mode)
        if self.initial_guess_list[self.initial_guess_index] is not None:
            self.initial_guess_density = self.get_initial_guess_density(
                gridpoints, self.comparison_mol, self.initial_guess_list[self.initial_guess_index]
            )
        if self.show_comparison_mol:
            ax2 = self.fig.add_subplot(fig_number + 2)
            if self.comparison_mol_has_orbitals:
                self.plot_kohn_sham(
                    self.fig,
                    ax2,
                    x_,
                    y_,
                    gridpoints,
                    self.comparison_mol,
                    self.gamma,
                    show_full_range,
                )
            else:
                self.plot_orbital_free_density(
                    self.fig,
                    ax2,
                    x_,
                    y_,
                    gridpoints,
                    self.comparison_mol,
                    self.of_coeffs,
                    show_full_range,
                )

        if self.recalc:
            n_ao = [mol.nao for mol in self.mol_with_density_basis]
            if "curvature" in self.current_mode:
                ao_p_derivs = [
                    np.transpose(
                        dft.numint.eval_ao(
                            mol, gridpoints.reshape(3, self.n_pixel * self.n_pixel).T, deriv=2
                        ),
                        (0, 2, 1),
                    ).reshape(10, n_ao[j], self.n_pixel, self.n_pixel)
                    for j, mol in enumerate(self.mol_with_density_basis)
                ]
                # 0,x,y,z,xx,xy,xz,yy,yz,zz
                ao_p = [
                    ao_p_derivs[i][4] + ao_p_derivs[i][7] + ao_p_derivs[i][9]
                    for i in range(len(self.mol_with_density_basis))
                ]
            else:
                ao_p = [
                    dft.numint.eval_ao(
                        mol, gridpoints.reshape(3, self.n_pixel * self.n_pixel).T
                    ).T.reshape(n_ao[j], self.n_pixel, self.n_pixel)
                    for j, mol in enumerate(self.mol_with_density_basis)
                ]
            self.densities_of = np.zeros((self.n_images, self.n_pixel, self.n_pixel))
        viewer_density_of = np.zeros((self.n_images, self.n_pixel, self.n_pixel))
        for i in range(self.n_images):
            if not self.show_mol_list and (
                self.of_coeffs_list[basis_index[i]][method_index[i]] is None
            ):
                continue
            newax = self.fig.add_subplot(fig_number + i + 2 + self.show_comparison_mol)
            plot_title = (
                self.density_fitting_methods[method_index[i]]
                + "|"
                + self.basis_sets[basis_index[i]]
            )
            if self.recalc:
                if self.mol_list_has_orbitals:
                    self.densities_of[i] = self.calcualte_density_from_orbitals(
                        gridpoints,
                        self.comparison_mols_with_orbital_basis[basis_index[i]],
                        self.comparison_mols_gamma_list[basis_index[i]],
                    )
                else:
                    self.densities_of[i] = self.calculate_of_density(
                        self.mol_with_density_basis[basis_index[i]],
                        self.of_coeffs_list[basis_index[i]][method_index[i]],
                        ao_p[basis_index[i].item()],
                    )
            if self.initial_guess_list[self.initial_guess_index] is not None:
                viewer_density_of[i] = self.densities_of[i] - self.initial_guess_density
            else:
                viewer_density_of[i] = self.densities_of[i]
            if "(ks-of)/of" in self.current_mode:
                residual_density = (self.density_ks - viewer_density_of[i]) / self.density_ks
                self.plot_contour(
                    self.fig,
                    newax,
                    x_,
                    y_,
                    residual_density,
                    plot_title,
                    [-self.max_range, +self.max_range, -self.max_range, +self.max_range],
                )
            elif "ks-of" in self.current_mode:
                residual_density = self.density_ks - viewer_density_of[i]
                self.plot_contour(
                    self.fig,
                    newax,
                    x_,
                    y_,
                    residual_density,
                    plot_title,
                    [-self.max_range, +self.max_range, -self.max_range, +self.max_range],
                )
            else:
                self.plot_contour(
                    self.fig,
                    newax,
                    x_,
                    y_,
                    viewer_density_of[i],
                    plot_title,
                    [-self.max_range, +self.max_range, -self.max_range, +self.max_range],
                )
        plt.show()

    def on_press(self, event):
        """Callback function to make the plot interactive."""
        self.mode_changed = False
        self.recalc = True
        print("press", event.key)
        sys.stdout.flush()
        plt.clf()
        if event.key == "y":
            self.offset[2] += self.scale * 1
        if event.key == "x":
            self.offset[2] += -self.scale * 1
        if event.key == "left":
            self.offset[0] += -self.scale * 1
        if event.key == "right":
            self.offset[0] += self.scale * 1
        if event.key == "up":
            self.offset[1] += self.scale * 1
        if event.key == "down":
            self.offset[1] += -self.scale * 1
        if event.key == "k":
            self.scale *= 2
        if event.key == "l":
            self.scale /= 2
        if event.key == "m":
            self.mode = (self.mode + 1) % len(self.mode_names)
            self.mode_changed = True
        if event.key == "n":
            self.show_mode = (self.show_mode + 1) % 2
        if event.key == "c":
            self.c_range *= 2
            self.recalc = False

        if event.key == "v":
            self.c_range /= 2
            self.recalc = False
        if event.key == "ö":
            self.index += 1
        if event.key == "ä":
            self.index -= 1
        if event.key == "b":
            self.labelmode = (self.labelmode + 1) % 3
        if event.key == "i":
            self.n_pixel *= 2
        if event.key == "o":
            self.n_pixel = self.n_pixel // 2
        if event.key == ",":
            self.bf_index += 1
        if event.key == ".":
            self.bf_index -= 1
        if event.key == "1":
            self.l_index += 1
        if event.key == "2":
            self.l_index -= 1
        if event.key == "w":
            phi = 0.2
            self.rot_mat = self.rot_mat @ np.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, np.cos(phi), np.sin(phi)),
                    (0.0, -np.sin(phi), np.cos(phi)),
                )
            )
        if event.key == "e":
            phi = -0.2
            self.rot_mat = self.rot_mat @ np.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, np.cos(phi), np.sin(phi)),
                    (0.0, -np.sin(phi), np.cos(phi)),
                )
            )
        if event.key == "r":
            phi = np.pi / 8
            self.rot_mat = self.rot_mat @ np.asarray(
                ((np.cos(phi), np.sin(phi), 0.0), (-np.sin(phi), np.cos(phi), 0), (0, 0, 1))
            )
        if event.key == "t":
            phi = -np.pi / 8
            self.rot_mat = self.rot_mat @ np.asarray(
                ((np.cos(phi), np.sin(phi), 0.0), (-np.sin(phi), np.cos(phi), 0), (0, 0, 1))
            )
        if event.key == "z":
            phi = np.pi / 8
            self.rot_mat = self.rot_mat @ np.asarray(
                ((np.cos(phi), 0.0, np.sin(phi)), (0.0, 1.0, 0.0), (-np.sin(phi), 0, np.cos(phi)))
            )
        if event.key == "u":
            phi = -np.pi / 8
            self.rot_mat = self.rot_mat @ np.asarray(
                ((np.cos(phi), 0.0, np.sin(phi)), (0.0, 1.0, 0.0), (-np.sin(phi), 0, np.cos(phi)))
            )
        if event.key == "8":
            self.initial_guess_index = (self.initial_guess_index + 1) % len(
                self.initial_guess_list
            )
        if event.key == "9":
            self.annote_atoms = (self.annote_atoms + 1) % 4
        if event.key == "3":
            self.shift_index += 1
        if event.key == "4":
            self.shift_index -= 1
        if self.from_density_fitting_statistic:
            if event.key == "d":
                self._mol_number_index = (self._mol_number_index + 1) % len(
                    self.dfs.unique_mol_numbers
                )
                self.mol_number = self.dfs.unique_mol_numbers[self._mol_number_index]
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)
            if event.key == "g":
                self._mol_number_index = (self._mol_number_index - 1) % len(
                    self.dfs.unique_mol_numbers
                )
                self.mol_number = self.dfs.unique_mol_numbers[self._mol_number_index]
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)
            if event.key == "j":
                self.scf_iter = (self.scf_iter + 1) % self.max_scf_iter
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)
            if event.key == "h":
                self.scf_iter = (self.scf_iter - 1) % self.max_scf_iter
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)
            if event.key == "p":
                if self.show_orbital is None:
                    self.show_orbital = -1
                self.show_orbital = self.show_orbital + 1
                if self.show_orbital == self.n_orbitals:
                    self.show_orbital = None
                self.mode_changed = True
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)
            if event.key == "ü":
                if self.show_orbital is None:
                    self.show_orbital = self.n_orbitals
                self.show_orbital = self.show_orbital - 1
                if self.show_orbital == -1:
                    self.show_orbital = None
                self.mode_changed = True
                self.load_density_fitting_statictics(self.dfs, self.mol_number, self.scf_iter)

        else:
            if event.key in ("d", "g", "j", "h", "p", "ü"):
                logger.warning(
                    "This option is only possible if this class was initialized from a Datasetsetatistics class"
                )
        self.plot_fig()

        self.fig.canvas.draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_difference_interactive(self):
        """Display an interactive window containing the visualisation, using the "TkAgg and connect
        event callbacks to the keys" backend."""
        if self.dark_mode:
            plt.style.use("dark_background")
        mpl.use("TkAgg")
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.plot_fig()

    def plot_difference(self):
        """Display a figure containing the visualisation, using the default backend."""
        if self.dark_mode:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.plot_fig()


def rotate_molecule_pca(mol_ks: gto.Mole) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a molecule onto the plane of most symmetry using PCA.

    Args:
        mol_ks (gto.Mole): the molecule we want to rotate
    Returns:
        offset (np.ndarray) The offset of the center of the molecule to the origin
        rotation matrix (np.ndarray) Rotation matrix to rotate the molecules symmetry axis onto the x-y plane
    """
    atom_pos = mol_ks.atom_coords()
    atom_charges = mol_ks.atom_charges()
    mask_heavy_atoms = atom_charges > 1
    heavy_atom_pos = atom_pos[mask_heavy_atoms]
    if heavy_atom_pos.shape[0] <= 2:
        heavy_atom_pos = atom_pos
    mean = np.mean(heavy_atom_pos, axis=0)
    std = np.std(heavy_atom_pos)
    normed_heavy_atom_pos = (heavy_atom_pos - mean) / std
    covariance_matrix = np.cov(normed_heavy_atom_pos, ddof=1, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1]

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    # use the first 3 to rotate the molecule
    matrix = sorted_eigenvectors[:3]

    return mean, matrix


def rotate_molecule2_onto_plane(mol_ks: gto.Mole) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a molecule onto the plane of most symmetry by just rotating the 3 heaviest atoms into
    a plane.

    Args:
        mol_ks (gto.Mole): the molecule we want to rotate
    Returns:
        offset (np.ndarray) The offset of the center of the molecule to the origin
        rotation matrix (np.ndarray) Rotation matrix to rotate the molecules symmetry axis onto the x-y plane
    """
    atom_pos = mol_ks.atom_coords()
    atom_charges = mol_ks.atom_charges()
    heavy_atoms_index = np.argsort(atom_charges)[::-1]
    big3 = atom_pos[heavy_atoms_index][:3]
    offset = big3[0]
    big3 -= big3[0]
    x_vec = big3[1]
    y_vec = big3[2]
    # Norm and orthogonalize them
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec - x_vec * (x_vec @ y_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = np.cross(x_vec, y_vec)

    mat = np.vstack((x_vec, y_vec, z_vec))
    return offset, mat.transpose()

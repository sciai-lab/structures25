import logging
from functools import partial
from typing import Any, Callable

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyscf import dft

from mldft.ml.callbacks.base import OnStepCallbackWithTiming
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.cube_files import DataCube
from mldft.utils.molecules import build_molecule_ofdata
from mldft.utils.visualize_3d import (
    ATOM_COLORS,
    find_isosurface_value,
    get_sticks_mesh_dict,
)

logger = logging.getLogger(__name__)


def hex_to_rgb(hex_color: str) -> tuple:
    """Converts a hex color string to an rgb tuple.

    Args:
        hex_color: The hex color string, e.g. ``#FF0000``.

    Returns:
        tuple: The rgb tuple, e.g. ``(255, 0, 0)``.
    """
    hex_color = hex_color.lstrip("#")
    h_len = len(hex_color)
    return tuple(int(hex_color[i : i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def combine_mesh_dicts(mesh_dicts: list[dict]) -> dict:
    """Combines multiple mesh dicts into one.

    Args:
        mesh_dicts: A list of mesh dicts. Each dict must have exactly the keys ``vertices``, ``faces``, ``colors``.

    Returns:
        dict: The combined mesh dict.
    """
    for mesh_dict in mesh_dicts:
        assert mesh_dict.keys() == {
            "vertices",
            "faces",
            "colors",
        }, f'Keys of mesh dicts must be "vertices", "faces", "colors". Got {mesh_dicts[0].keys()}'

    vertex_ind_offsets = np.cumsum(
        [0] + [mesh_dict["vertices"].shape[1] for mesh_dict in mesh_dicts]
    )
    vertices = np.concatenate([mesh_dict["vertices"] for mesh_dict in mesh_dicts], axis=1)
    faces = np.concatenate(
        [
            vertex_ind_offset + mesh_dict["faces"]
            for mesh_dict, vertex_ind_offset in zip(mesh_dicts, vertex_ind_offsets)
        ],
        axis=1,
    )
    colors = np.concatenate([mesh_dict["colors"] for mesh_dict in mesh_dicts], axis=1)

    return dict(vertices=vertices, faces=faces, colors=colors)


class LogMolecule(OnStepCallbackWithTiming):
    """Logs the molecule to tensorboard as a mesh."""

    def __init__(
        self,
        max_molecule_resolution: int = 10,
        max_isosurface_resolution: float = 0.2,
        basis_transformation: str | Callable | None = "auto",
        log_initial_guess: bool = True,
        log_gradient: bool = True,
        log_random_basis_functions: bool = False,
        **super_kwargs,
    ) -> None:
        """
        Args:
            max_molecule_resolution: The maximum "resolution" of the sticks representation of the molecule.
            max_isosurface_resolution: The maximum grid resolution used to compute the isosurface, in Bohr.
            basis_transformation: The basis transformation to be applied to the molecule.
                Can be either ``None``, 'auto', or a callable, e.g. a value of :const:`BASIS_TRANSFORMATIONS`.
                If 'auto', the basis transformation is chosen based on the label_subdir of the datamodule.
                ``None`` means no transformation is applied. Defaults to 'auto'.
            log_initial_guess: Whether to log the initial guess. Defaults to True.
            log_gradient: Whether to log the gradient. Defaults to True.
            log_random_basis_functions: Whether to log some random basis functions. Defaults to False.
                See :meth:`LogMolecule.get_mesh_kwargs` for details.
            **super_kwargs: Keyword arguments passed to :class:`OnStepCallbackWithTiming`.
        """
        super().__init__(**super_kwargs)

        self.max_molecule_resolution = max_molecule_resolution
        self.max_isosurface_resolution = max_isosurface_resolution
        self.basis_transformation = basis_transformation

        self.log_initial_guess = log_initial_guess
        self.log_gradient = log_gradient
        self.log_random_basis_functions = log_random_basis_functions

        # Camera and scene configuration.
        # for documentation, see https://threejs.org/docs
        self.config_dict = {
            # use a relatively small fov in order to be able to tell when things are parallel / perpendicular
            "camera": {"cls": "PerspectiveCamera", "fov": 10},
            "lights": [
                {
                    "cls": "AmbientLight",
                    "color": "#ffffff",
                    "intensity": 0.75,
                },
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.75,
                    "position": [0, -1, 2],
                },
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.5,
                    "position": [0, 1, -2],
                },
            ],
            "material": {
                "cls": "MeshStandardMaterial",
                "roughness": 0.4,
                "metalness": 0.0,
                "transparent": True,
                "opacity": 0.75,
            },
        }

    def get_molecule_dict(self, mol: Any) -> dict:
        """Get a sticks representation of the molecule as a mesh dict."""
        # get the sticks mesh, adapting the resolution to get at most 10000 points
        resolution = self.max_molecule_resolution
        sticks_mesh_dict = get_sticks_mesh_dict(mol, resolution=resolution)
        while resolution > 1 and sticks_mesh_dict["mesh"].n_points > 10000:
            resolution //= 2
            sticks_mesh_dict = get_sticks_mesh_dict(mol, resolution=resolution)

        mesh = sticks_mesh_dict["mesh"]
        mesh = mesh.triangulate()
        assert mesh.is_all_triangles, "Need a mesh with all triangles"
        vertices = mesh.points[None]
        # first index is the number of vertices per face, here always 3
        faces = mesh.faces.reshape((-1, 4))[None, :, 1:]

        assert np.allclose(
            mesh["color_ids"], np.round(mesh["color_ids"])
        ), "color_ids must be integers"
        atom_colors = np.array(
            [hex_to_rgb(hex_color) for hex_color in ATOM_COLORS.values()], dtype=np.int32
        )
        face_vertex_colors = atom_colors[None, np.round(mesh["color_ids"]).astype(np.int32)]

        # the colors are one per each occurrence of the vertex (so multiple if it appears in multiple faces)
        # we need a single color per vertex, so we take an arbitrary one.
        # if a vertex is not in any face, it will be black
        colors = np.zeros((1, vertices.shape[1], 3), dtype=np.int32)
        for i in range(3):  # iterate over the indices of the corners of (all) the triangles
            colors[:, faces[0, :, i]] = face_vertex_colors

        return dict(
            vertices=vertices.copy(),
            faces=faces.copy(),
            colors=colors.copy(),
        )

    def get_mesh_kwargs(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        batch: Any,
        outputs: STEP_OUTPUT,
        basis_info: BasisInfo,
    ) -> dict:
        """Create a dictionary of meshes to be logged.

        Args:
            trainer: The lightning trainer.
            pl_module: The lightning module.
            batch: The batch.
            outputs: The outputs of the lightning module.
            basis_info: The :class:`BasisInfo` object.

        Returns:
            dict: A dict of dicts where each value is a dict with arguments to ``SummaryWriter.add_mesh``.
        """
        result_dict = {}
        logging_keys = []

        sample = batch.to_data_list()[0].detach().cpu()
        transform = trainer.val_dataloaders.dataset.transforms
        # construct the molecule
        mol = build_molecule_ofdata(ofdata=sample, basis=basis_info.basis_dict)

        try:
            mol_mesh_dict = self.get_molecule_dict(mol)
        except Exception as e:
            # this can happen e.g. if rdkit is unhappy with the molecule
            logger.error(f"Could not build molecule: {e}")
            return dict()

        mol_mesh_dict_with_scf_iter = mol_mesh_dict.copy()
        mol_mesh_dict_with_scf_iter["scalars"] = dict(scf_iteration=sample.scf_iteration.item())

        if self.log_initial_guess:
            # this works only for the first sample!
            pred_coeffs = (
                outputs["model_outputs"]["pred_diff"][: len(sample.coeffs)].detach().cpu()
            )
            gt_coeffs = sample.coeffs - sample.ground_state_coeffs
            coeff_diff = pred_coeffs - gt_coeffs
            result_dict["initial_guess/0_mol"] = mol_mesh_dict_with_scf_iter
            sample.add_item("initial_guess/1_gt", gt_coeffs, Representation.VECTOR)
            sample.add_item("initial_guess/2_pred", pred_coeffs, Representation.VECTOR)
            sample.add_item("initial_guess/3_diff", coeff_diff, Representation.VECTOR)
            logging_keys.extend(
                ["initial_guess/1_gt", "initial_guess/2_pred", "initial_guess/3_diff"]
            )

        if self.log_gradient:
            gradient_label = sample.gradient_label.detach().cpu()
            pred_gradients = (
                outputs["model_outputs"]["pred_gradients"][: len(sample.coeffs)].detach().cpu()
            )
            projected_gradient_difference = (
                outputs["projected_gradient_difference"][: len(sample.coeffs)].detach().cpu()
            )
            sample.add_item("gradient/1_gt", gradient_label, Representation.GRADIENT)
            sample.add_item("gradient/2_pred", pred_gradients, Representation.GRADIENT)
            sample.add_item(
                "gradient/3_diff",
                projected_gradient_difference,
                Representation.GRADIENT,
            )
            logging_keys.extend(["gradient/1_gt", "gradient/2_pred", "gradient/3_diff"])
            result_dict["gradient/0_mol"] = mol_mesh_dict_with_scf_iter

        if self.log_random_basis_functions:
            basis_func_to_l = basis_info.l_per_basis_func[sample.basis_function_ind.cpu().numpy()]
            basis_func_to_m = basis_info.m_per_basis_func[sample.basis_function_ind.cpu().numpy()]

            logging_keys = []
            # plot some random s, l, and d functions, to check that their orientation is correct
            for i, (l, n) in enumerate(((0, 1), (1, 1), (1, 3), (1, 3), (2, 2), (3, 2), (4, 2))):
                debug_local_frames_coeffs = np.zeros_like(sample.coeffs)
                ind = np.random.choice(
                    np.argwhere(np.bitwise_and(basis_func_to_l == l, basis_func_to_m == 0))[
                        :, 0
                    ],  # only m == 0
                    # np.argwhere(basis_func_to_l == l)[:, 0],  # random m
                    size=n,
                    replace=False,
                )
                debug_local_frames_coeffs[ind] = 1
                debug_local_frames_coeffs_torch = torch.as_tensor(debug_local_frames_coeffs)
                sample.add_item(
                    f"basis_functions/{i}",
                    debug_local_frames_coeffs_torch,
                    Representation.VECTOR,
                )
                logging_keys.append(f"basis_functions/{i}")

        def get_value(coords: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
            """Evaluate the value to compute the isosurfaces from on a given array of coords (shape
            (N, 3))."""
            ao = dft.numint.eval_ao(mol, coords, deriv=0)
            return np.dot(ao, coeffs)

        sample = transform.invert_basis_transform(sample)
        for key in logging_keys:
            coeffs = sample[key].float().numpy()
            resolution = self.max_isosurface_resolution
            while True:
                # 1. compute the value on the grid
                cube = DataCube.from_function(
                    mol=mol,
                    func=partial(get_value, coeffs=coeffs),
                    resolution=resolution,
                )
                pyvista_grid = cube.to_pyvista()

                # 2. compute the isosurface
                quantiles = np.array([0.5])  # hardcoded for now
                isosurface_value = find_isosurface_value(pyvista_grid["data"], quantiles, p=1)
                # add negative isosurface values
                isosurface_values = np.concatenate([-isosurface_value, isosurface_value])

                isosurface = pyvista_grid.contour(isosurfaces=isosurface_values)
                iso_mesh = isosurface.extract_geometry()
                if iso_mesh.n_points <= 50000:
                    break
                else:
                    resolution *= 2

            if iso_mesh.n_points > 0:
                iso_mesh_colors = np.empty((iso_mesh.n_points, 3), dtype=np.int32)

                # assign colors based on isosurface value
                positive_mask = iso_mesh["data"] > 0
                iso_mesh_colors[positive_mask] = np.array([255, 0, 0])  # red
                iso_mesh_colors[~positive_mask] = np.array([0, 0, 255])  # blue

                # orient the negative faces correctly (otherwise they are see-through from the outside)
                iso_faces = iso_mesh.faces.reshape((-1, 4))[:, 1:].copy()
                negative_face_mask = iso_mesh["data"][iso_faces[:, 0]] < 0
                iso_faces[negative_face_mask] = iso_faces[negative_face_mask][:, ::-1]

                # 3. add the isosurface to the faces, vertices, corners

                mesh_dict = combine_mesh_dicts(
                    [
                        mol_mesh_dict,
                        dict(
                            vertices=iso_mesh.points[None],
                            faces=iso_faces[None],
                            colors=iso_mesh_colors[None],
                        ),
                    ]
                )
                mesh_dict["scalars"] = dict(isovalue=float(isosurface_value[0]))
            else:
                logger.debug("No isosurface found")
                mesh_dict = mol_mesh_dict

            result_dict[f"_{key}"] = mesh_dict

        return result_dict

    def execute(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        split: str,
    ) -> None:
        """Logs a mesh to tensorboard at the end of a training batch, if the timing matches."""

        logger.debug(f"Logging training mesh to tensorboard from {self.__class__.__name__}")

        basis_info = pl_module.basis_info

        tb_logger = pl_module.tensorboard_logger
        for key, mesh_dict in self.get_mesh_kwargs(
            trainer, pl_module, batch, outputs, basis_info
        ).items():
            if "scalars" in mesh_dict:
                for scalar_name, scalar_value in mesh_dict.pop("scalars").items():
                    tb_logger.add_scalar(
                        # add "y_" for ordering of tags in tensorboard
                        # (below everything but logging mixins, which start with "z_")
                        tag=f"y_mesh_{split}_{self.name}{key}_{scalar_name}",
                        scalar_value=scalar_value,
                        global_step=trainer.global_step,
                    )
            tb_logger.add_mesh(
                f"{split}_{self.name}{key}",
                **mesh_dict,
                config_dict=self.config_dict,
                global_step=trainer.global_step,
            )

import warnings
from typing import IO

import numpy as np
import pyscf
from pyscf import gto
from pyscf.tools import cubegen


class DataCube:
    """A data cube, containing a field on a cartesian grid, spanning a molecule."""

    def __init__(self, mol: pyscf.gto.Mole, box: np.ndarray, origin: np.ndarray, data: np.ndarray):
        """Initialize a DataCube.

        Args:
            mol: The pyscf molecule.
            box: The vectors spanning the box, defining x y and z directions.
            origin: The origin of the box.
            data: The data, shape (nx, ny, nz).
        """
        self.mol = mol
        self.box = box
        self.origin = origin
        self.data = data

    def __str__(self):
        return f"DataCube(mol={self.mol}, box={self.box}, origin={self.origin}, data.shape={self.data.shape})"

    @classmethod
    def from_file(cls, filename: str | IO, is_tiling_unit_cell=False) -> "DataCube":
        """Read a cube file. Adapted from `pyscf.tools.cubegen.Cube.read`.
        For details on the format see
        https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html#cubeformat-dset-ids

        Args:
            filename: The filename.
            is_tiling_unit_cell: Whether to use an asymmetric mesh for tiling unit cells.

        Returns:
            A DataCube object.
        """
        with open(filename) as f:
            return cls.from_fileobject(f, is_tiling_unit_cell)

    @classmethod
    def from_fileobject(cls, f: IO, is_tiling_unit_cell=False):
        """Read a cube file. Adapted from `pyscf.tools.cubegen.Cube.read`.
        For details on the format see
        https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html#cubeformat-dset-ids

        Args:
            f: The file object.
            is_tiling_unit_cell: Whether to use an asymmetric mesh for tiling unit cells.

        Returns:
            A DataCube object.
        """
        f.readline()
        f.readline()
        data = f.readline().split()
        natm = int(data[0])
        has_dset_ids = natm < 0  # this indicates another line after molecule description
        natm = np.abs(natm)

        box_origin = np.array([float(x) for x in data[1:]])

        def parse_nx(data):
            d = data.split()
            nx = int(d[0])
            x_vec = np.array([float(x) for x in d[1:]]) * nx
            if is_tiling_unit_cell:
                # Use an asymmetric mesh for tiling unit cells
                xs = np.linspace(0, 1, nx, endpoint=False)
            else:
                # Use endpoint=True to get a symmetric mesh
                # see also the discussion https://github.com/sunqm/pyscf/issues/154
                xs = np.linspace(0, 1, nx, endpoint=True)
            return x_vec, nx, xs

        # get box dimensions and resolution
        box = np.zeros((3, 3))
        box[0], nx, xs = parse_nx(f.readline())
        box[1], ny, ys = parse_nx(f.readline())
        box[2], nz, zs = parse_nx(f.readline())

        # construct the molecule
        atoms = []
        for ia in range(natm):
            d = f.readline().split()
            atoms.append([int(d[0]), [float(x) for x in d[2:]]])
        mol = gto.M(atom=atoms, unit="Bohr")

        if has_dset_ids:
            warnings.warn("Cube file has DSET_IDS. Parsing these is not implemented, ignoring.")
            f.readline()

        # read and reshape cube data
        data = f.read()
        cube_data = np.array([float(x) for x in data.split()])
        assert nx * ny * nz == len(cube_data), f"{nx*ny*nz=} != {len(cube_data)}"
        cube_data = cube_data.reshape([nx, ny, nz])

        return cls(
            mol=mol,
            box=box,
            origin=box_origin,
            data=cube_data,
        )

    @classmethod
    def from_function(
        cls,
        mol: pyscf.gto.Mole,
        func: callable,
        resolution: float = cubegen.RESOLUTION,
        margin: float = cubegen.BOX_MARGIN,
        block_size: int = 16384,
    ):
        """Create a DataCube by evaluating a given function on a cartesian grid spanning a
        molecule.

        Args:
            mol: The pyscf molecule.
            func: The function to evaluate on the grid. Must take an array of positions of shape (n, 3) as input.
            resolution: The resolution of the grid in Bohr.
            margin: The margin of the box.
            block_size: The blocksize for the evaluation. Defaults to 16384.
        """

        cc = cubegen.Cube(mol, nx=None, ny=None, nz=None, resolution=resolution, margin=margin)
        coords = cc.get_coords()
        ngrids = cc.get_ngrids()
        block_size = ngrids if block_size is None else block_size
        data = np.empty(ngrids)
        for ip0, ip1 in pyscf.lib.prange(0, ngrids, block_size):
            data[ip0:ip1] = func(coords[ip0:ip1])
        data = data.reshape(cc.nx, cc.ny, cc.nz)

        # off by one bs
        box = np.array([b * s / (s - 1) for b, s in zip(cc.box, data.shape)])

        return cls(mol, box, cc.boxorig, data)

    def to_pyvista(self):
        """Convert to a pyvista grid.

        Args:
            self: The DataCube.

        Returns:
            The pyvista grid.
        """

        import pyvista as pv

        assert np.max(np.abs(self.box[~np.eye(3, dtype=bool)])) == 0, "Box must be diagonal."

        grid = pv.ImageData(
            dimensions=self.data.shape,
            origin=self.origin,
            spacing=[self.box[i, i] / self.data.shape[i] for i in range(3)],
        )
        grid.point_data["data"] = self.data.ravel(order="F")

        return grid

    # def to_data_string(self, field, fname, comment=None):
    #
    #     result = StringIO()
    #     cubegen.Cube.write()
    #

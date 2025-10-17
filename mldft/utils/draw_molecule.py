import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdDetermineBonds


def load_molecule(path: Path) -> Chem.Mol:
    """Load a molecule from a .xyz or .sdf file.

    In case of xyz files the bond orders are tried to be determined automatically.

    Args:
        path (Path): Path to the file

    Returns:
        Chem.Mol: RDKit molecule object
    """
    if path.suffix == ".sdf":
        mol = Chem.MolFromMolFile(str(path))
    elif path.suffix == ".xyz":
        xyz_block = cleanup_xyz(path)
        mol = Chem.MolFromXYZBlock(xyz_block)
        rdDetermineBonds.DetermineBonds(mol)
        rdDetermineBonds.DetermineBondOrders(mol)
    else:
        raise ValueError(
            f"Unknown file type {path.suffix}, only .xyz and .sdf are currently implemented"
        )

    return mol


def cleanup_via_smiles(mol: Chem.Mol) -> Chem.Mol:
    """Convert a molecule to a smiles string and back to a molecule to clean it up for drawing.

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        Chem.Mol: Cleaned up RDKit molecule object
    """
    smiles_str = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles_str)
    return mol


# This function is especially required for the QM9 xyz files as they contain additional information
# which rdkit cannot handle.
def cleanup_xyz(path: Path) -> str:
    """Read an xyz file and return a cleaned up version of it which rdkit can handle.

    Args:
        path (Path): Path to the xyz file
    Returns:
        str: Cleaned up xyz block
    """
    fixed_xyz = ""
    with open(path) as f:
        lines = f.readlines()
        n_atoms = int(lines[0])
        lines = lines[2 : n_atoms + 2]
        fixed_xyz += f"{n_atoms}\n\n"
        for line in lines:
            atom, x, y, z = line.split()[:4]
            fixed_xyz += f"{atom} {x} {y} {z}\n"

    return fixed_xyz


def draw_molecule(mol: Chem.Mol, width: int, height: int) -> str:
    """Draw a molecule to an SVG image.

    Args:
        mol (Chem.Mol): RDKit molecule object
        width (int): Width of the image
        height (int): Height of the image

    Returns:
        str: SVG image
    """
    img = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = img.drawOptions()
    opts.updateAtomPalette({i: (0, 0, 0) for i in range(1, 118)})
    img.DrawMolecule(
        mol,
    )
    img.FinishDrawing()

    svg = img.GetDrawingText()

    return svg


def set_general_options() -> None:
    """Set general options for the RDKit.

    Currently just a stub for setting the preferCoordGen option and might be later expanded.
    """
    rdDepictor.SetPreferCoordGen(True)


def write_file(svg: str, path: Path) -> None:
    """Write an SVG image to a file.

    Args:
        svg (str): SVG image
        path (Path): Path to the output file
    """
    with open(path, "w") as f:
        f.write(svg)


def parse_args(args: list) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to the input file")
    parser.add_argument("-o", "--output", type=Path, help="Path to the output file", default=None)
    parser.add_argument("--width", type=int, default=500, help="Width of the output image")
    parser.add_argument("--height", type=int, default=500, help="Height of the output image")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    set_general_options()

    mol = load_molecule(args.input)
    mol = cleanup_via_smiles(mol)
    svg = draw_molecule(mol, args.width, args.height)

    if args.output is None:
        args.output = Path.cwd() / f"{args.input.stem}.svg"

    write_file(svg, args.output)


if __name__ == "__main__":
    main()

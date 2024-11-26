import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

alkali_metals = [3, 11, 19, 37, 55, 87]  # Li, Na, K, Rb, Cs, Fr
alkaline_earth_metals = [4, 12, 20, 38, 56, 88]  # Be, Mg, Ca, Sr, Ba, Ra
other_metals = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]  # Sc to Zn, etc.


def adjust_charges(atom, neighbor):
    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    neighbor.SetFormalCharge(neighbor.GetFormalCharge() - 1)


def break_bonds_with_heteroatoms(mol):
    emol = Chem.EditableMol(mol)
    metals = alkali_metals + alkaline_earth_metals
    bonds_to_remove = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if bond.GetBondType() == Chem.BondType.SINGLE:
            if atom1.GetAtomicNum() in metals and atom2.GetAtomicNum() not in metals + [6]:  
                bonds_to_remove.append((atom1.GetIdx(), atom2.GetIdx()))
                adjust_charges(atom1, atom2)
            elif atom2.GetAtomicNum() in metals and atom1.GetAtomicNum() not in metals + [6]:
                bonds_to_remove.append((atom1.GetIdx(), atom2.GetIdx()))
                adjust_charges(atom2, atom1)

    for bond in bonds_to_remove:
        emol.RemoveBond(bond[0], bond[1])

    return emol.GetMol()


def replace_bonds_with_dative(mol):
    emol = Chem.EditableMol(mol)
    bonds_to_remove = []
    bonds_to_add = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if bond.GetBondType() == Chem.BondType.SINGLE:
            if atom1.GetAtomicNum() in other_metals and atom2.GetAtomicNum() not in other_metals:
                bonds_to_remove.append((atom1.GetIdx(), atom2.GetIdx()))
                bonds_to_add.append((atom2.GetIdx(), atom1.GetIdx()))
            elif atom2.GetAtomicNum() in other_metals and atom1.GetAtomicNum() not in other_metals:
                bonds_to_remove.append((atom1.GetIdx(), atom2.GetIdx()))
                bonds_to_add.append((atom1.GetIdx(), atom2.GetIdx()))

    for bond in bonds_to_remove:
        emol.RemoveBond(bond[0], bond[1])
    for bond in bonds_to_add:
        emol.AddBond(bond[0], bond[1], Chem.BondType.DATIVE)

    return emol.GetMol()


def correct_metallocenes(mol):
    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in other_metals:
                neighbors = atom.GetNeighbors()
                aromatic_rings = []
                for nbr in neighbors:
                    if nbr.GetIsAromatic():
                        aromatic_rings.append(nbr.GetIdx())

                if len(aromatic_rings) == 2:
                    AllChem.Kekulize(mol, clearAromaticFlags=True)
                    for ring_atom_idx in aromatic_rings:
                        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
                        mol.AddBond(ring_atom.GetIdx(), atom.GetIdx(), Chem.BondType.DATIVE)
                    atom.SetFormalCharge(0)

        return mol
    except Exception as e:
        raise ValueError(f"Error correcting metallocene: {e}")


def process_molecule(mol):
    try:
        if mol is None:
            raise ValueError("Invalid molecule")

        mol = Chem.AddHs(mol)
        mol = break_bonds_with_heteroatoms(mol)
        mol = replace_bonds_with_dative(mol)
        mol = correct_metallocenes(mol)

        return mol, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Process molecules from an SDF file to handle metal bonds.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    error_file = args.error

    suppl = Chem.SDMolSupplier(input_file)
    writer = SDWriter(output_file)

    with open(error_file, 'w') as errfile:
        errfile.write("Original_SMILES,Error\n")

        for mol in suppl:
            if mol is None:
                continue  # Skip invalid molecules
            original_smiles = Chem.MolToSmiles(mol)
            processed_mol, error = process_molecule(mol)

            if processed_mol:
                writer.write(processed_mol)
                processed_mol.SetProp("metal_error", "None")
            else:
                errfile.write(f"{original_smiles},{error}\n")

    writer.close()


if __name__ == "__main__":
    main()


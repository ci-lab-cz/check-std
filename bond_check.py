import argparse
from rdkit import Chem


def process_and_fix_bonds(input_sdf):
   
    supplier = Chem.SDMolSupplier(input_sdf, sanitize = False)
    fixed_mols = []
    error_mols = []

    for mol in supplier:
        error_message = None

        if mol is None:
            error_message = "Error: Invalid molecule block."
        else:
            try:
                for bond in mol.GetBonds():
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        stereo = bond.GetStereo()
                        if stereo == Chem.BondStereo.STEREOANY:
                            try:
                                bond.SetStereo(Chem.BondStereo.STEREONONE)
                            except Exception as e:
                                error_message = f"Error: Failed to fix undefined double bond stereochemistry: {e}"

                for atom in mol.GetAtoms():
                    for bond in atom.GetBonds():
                        if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
                            if not atom.GetChiralTag():
                                try:
                                    bond.SetBondDir(Chem.BondDir.NONE)
                                except Exception as e:
                                    error_message = f"Error: Failed to clear invalid wedge bond: {e}"

                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

            except Exception as e:
                error_message = f"Error during bond revaluation: {str(e)}"

        if error_message:
            if mol:
                mol.SetProp("Error", error_message)
            error_mols.append(mol)
        else:
            if mol:
                mol.SetProp("Error", "None")
            fixed_mols.append(mol)

    return fixed_mols, error_mols


def rebuild_sdf(molecules, output_filename):
    """
    Rebuild the SDF file with processed molecules.
    """
    with Chem.SDWriter(output_filename) as writer:
        for mol in molecules:
            if mol is not None:
                writer.write(mol)


def main():
    parser = argparse.ArgumentParser(description="Process SDF files for bond revaluation.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    args = parser.parse_args()
    fixed_mols, error_mols = process_and_fix_bonds(args.input)
    rebuild_sdf(fixed_mols, args.output)
    rebuild_sdf(error_mols, args.error)


if __name__ == "__main__":
    main()


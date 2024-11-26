import argparse
from rdkit import Chem

def assign_charges_based_on_elements(mol):

    metals = ['Na', 'K', 'Li', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Ag', 'Au']
    halides = ['Cl', 'Br', 'I', 'F']
    bonds_to_remove = []  
    charge_error = None

    try:
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_symbol = begin_atom.GetSymbol()
            end_symbol = end_atom.GetSymbol()

            if (begin_symbol in metals and end_symbol in halides) or \
               (end_symbol in metals and begin_symbol in halides):
                if begin_symbol in metals:
                    begin_atom.SetFormalCharge(+1)
                    end_atom.SetFormalCharge(-1)
                else:
                    begin_atom.SetFormalCharge(-1)
                    end_atom.SetFormalCharge(+1)

                bonds_to_remove.append(bond.GetIdx())
        editable_mol = Chem.RWMol(mol)
        for bond_idx in bonds_to_remove:
            editable_mol.RemoveBond(
                mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
            )
        return editable_mol.GetMol(), None  # No error
    except Exception as e:
        charge_error = f"Error during charge assignment: {e}"
        return mol, charge_error

def detect_and_fix_stereochemistry(mol):
    """
    Detect and fix undefined stereochemistry in the molecule.
    """
    undefined_stereo_atoms = []
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED and \
               not atom.HasProp('_CIPCode'):
                undefined_stereo_atoms.append(atom.GetIdx())

        # Clear undefined stereochemistry
        for atom_idx in undefined_stereo_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

        if undefined_stereo_atoms:
            return mol, f"Cleared undefined stereo centers: {undefined_stereo_atoms}"
        return mol, None
    except Exception as e:
        return mol, f"Error during stereochemistry processing: {e}"

def sanitize_and_validate_kekulization(mol):
    """
    Perform final sanitization and validate kekulization.
    """
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)

        return mol, None  
    except Chem.KekulizeException:
        return mol, "Error: Can't kekulize molecule."
    except Exception as e:
        return mol, f"Sanitization or kekulization error: {e}"

def process_sdf(input_sdf, output_sdf, error_sdf):
    supplier = Chem.SDMolSupplier(input_sdf, sanitize=False)  
    writer = Chem.SDWriter(output_sdf)
    error_writer = Chem.SDWriter(error_sdf)

    for mol in supplier:
        if mol is None:
            print("Error reading molecule")
            continue

        mol.SetProp("Processing_Error", "")  

        try:
            # Step 1: Assign charges based on elements
            mol, charge_error = assign_charges_based_on_elements(mol)
            if charge_error:
                mol.SetProp("Processing_Error", charge_error)
                error_writer.write(mol)  
                continue

            mol, stereo_error = detect_and_fix_stereochemistry(mol)
            if stereo_error:
                mol.SetProp("Processing_Error", stereo_error)
                error_writer.write(mol)  # Save to error file
                continue

            mol, sanitization_error = sanitize_and_validate_kekulization(mol)
            if sanitization_error:
                mol.SetProp("Processing_Error", sanitization_error)
                error_writer.write(mol)  # Save to error file
                continue
            mol.SetProp("Processing_Error", "None")
            writer.write(mol)
        except Exception as e:
            error_message = f"Unexpected processing error: {e}"
            mol.SetProp("Processing_Error", error_message)
            error_writer.write(mol)  # Save to error file

    writer.close()
    error_writer.close()

def main():
    parser = argparse.ArgumentParser(description="Assign charges to molecules and process SDF files.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)

    args = parser.parse_args()

    input_sdf = args.input
    output_sdf = args.output
    error_sdf = args.error

    process_sdf(input_sdf, output_sdf, error_sdf)
    print(f"Processed SDF saved to {output_sdf}")
    print(f"Error SDF saved to {error_sdf}")

if __name__ == "__main__":
    main()


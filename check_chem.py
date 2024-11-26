import argparse
from rdkit import Chem
from collections import OrderedDict


def read_sdf_with_sdmolsupplier(input_filename):
    supplier = Chem.SDMolSupplier(input_filename, sanitize=False)
    molecules = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            # Capture invalid molecules
            molecules.append((None, None, {"ParsingError": f"Invalid molecule block at index {idx}"}))
        else:
            try:
                lem_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Unknown_LEM_ID_{idx + 1}"
                if mol.HasProp("LEM"):
                    mol.ClearProp("LEM")
                molecules.append((mol, lem_id, {}))  
            except Exception as e:
                molecules.append((mol, f"Unknown_LEM_ID_{idx + 1}", {"ParsingError": f"Error: {str(e)}"}))
    return molecules



def check_molecule(mol):
    errors = {}

    if mol is None or mol.GetNumAtoms() == 0:
        errors["CoordinateError"] = "No valid atoms or coordinates"
    else:
        try:
            problem_list = Chem.DetectChemistryProblems(mol)
            for problem in problem_list:
                if "valence" in problem.GetType().lower():
                    errors.setdefault("ValencyError", []).append(problem.Message())
        except Exception as e:
            errors["ValencyError"] = [f"Processing Error: {str(e)}"]

    for key, value in errors.items():
        if isinstance(value, list):
            errors[key] = "; ".join(value)

    return errors


def remove_duplicate_keep_first(molecules):
    unique_molecules = OrderedDict()
    for mol, lem_id, errors in molecules:
        if lem_id not in unique_molecules:
            unique_molecules[lem_id] = (mol, lem_id, errors)
    return list(unique_molecules.values())


def write_sdf(molecules, output_filename):
    """Write the molecules to an SDF file."""
    with Chem.SDWriter(output_filename) as writer:
        for mol, lem_id, errors in molecules:
            # Skip empty or invalid molecules
            if mol is None or mol.GetNumAtoms() == 0:
                continue

            if lem_id is None:
                lem_id = "Unknown_LEM_ID" 
            mol.SetProp("LEM_ID", lem_id)

            if not errors:
                mol.SetProp("Error_Message", "None")
            else:
                for error_field, error_message in errors.items():
                    mol.SetProp(error_field, error_message)

            writer.write(mol)

def process_sdf(input_filename, output_filename, error_filename):
    """Process the SDF file and separate valid and error molecules."""
    all_molecules = read_sdf_with_sdmolsupplier(input_filename)
    valid_molecules = []
    error_molecules = []

    for mol, lem_id, errors in all_molecules:
        if errors or mol is None or mol.GetNumAtoms() == 0:
            # If there are parsing errors, invalid molecules, or empty molecules
            error_molecules.append((mol, lem_id, errors))
        else:
            molecule_errors = check_molecule(mol)
            if molecule_errors:
                error_molecules.append((mol, lem_id, molecule_errors))
            else:
                valid_molecules.append((mol, lem_id, {}))
    all_molecules = valid_molecules + error_molecules
    unique_molecules = remove_duplicate_keep_first(all_molecules)

    valid_molecules = [(mol, lem_id, errors) for mol, lem_id, errors in unique_molecules if not errors]
    error_molecules = [(mol, lem_id, errors) for mol, lem_id, errors in unique_molecules if errors]

    write_sdf(valid_molecules, output_filename)
    write_sdf(error_molecules, error_filename)


def main():
    parser = argparse.ArgumentParser(description="Process SDF files and separate valid and error molecules.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file for valid molecules")
    parser.add_argument("-e", "--error", required=True, help="Output SDF file for error molecules")
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output
    error_filename = args.error

    process_sdf(input_filename, output_filename, error_filename)


if __name__ == "__main__":
    main()


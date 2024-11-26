#!/usr/bin/env python3

import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
#RDLogger.DisableLog('rdApp.*')

def neutralize_molecule(mol):
    """Neutralize charges in the molecule."""
    pattern_replacements = [
        ('[O-]', 'O'),   # Example: neutralize single negative oxygens
        ('[N+]', 'N')    # Example: neutralize single positive nitrogens
    ]
    for smarts, replacement in pattern_replacements:
        patt = Chem.MolFromSmarts(smarts)
        while mol.HasSubstructMatch(patt):
            rms = Chem.ReplaceSubstructs(mol, patt, Chem.MolFromSmiles(replacement), True)
            mol = rms[0]
    return mol

def strip_salts(mol):
    """Remove specified salts and ions from the molecule."""
    salts_smarts = [
        '[H][#8][H]', 'Cl[H]', 'Br[H]', 'I[H]', 'F[H]', '[Na+]', '[Ca++]', '[Cl-]', '[Br-]', '[I-]', 
        '[K+]', '[O]=[S](=O)O', 'C(=O)([O-])[O]', '[O-]C(=O)C(F)(F)F'
    ]
    for smarts in salts_smarts:
        patt = Chem.MolFromSmarts(smarts)
        while mol.HasSubstructMatch(patt):
            rms = Chem.DeleteSubstructs(mol, patt)
            mol = rms
    return mol

def handle_stereo_issues(mol):
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception as e:
        mol.SetProp("_Error", "Stereo issue: {}".format(str(e)))
    return mol
    
def clean_2d_structure(mol):
    """Generate 2D coordinates for the molecule."""
    AllChem.Compute2DCoords(mol)
    return mol

def standardize_structure(input_file: str, output_file: str, error_file: str):
    suppl = Chem.SDMolSupplier(input_file, sanitize=False, removeHs=False)  # Disable sanitize to load all molecules
    writer = Chem.SDWriter(output_file)
    error_log = []
    total_count = 0
    processed_count = 0
    error_count = 0

    for idx, mol in enumerate(suppl):
        total_count += 1
        if mol is None:
            error_message = "Molecule {}: Unreadable molecule".format(idx + 1)
            error_log.append(error_message)
            error_count += 1
            continue

        try:
            mol.SetProp("_Error", "None")
            Chem.SanitizeMol(mol)  # Sanitize the molecule
            mol = handle_stereo_issues(mol)
            mol = neutralize_molecule(mol)
            mol = strip_salts(mol)
            mol = clean_2d_structure(mol)
            writer.write(mol)
            processed_count += 1
        except Exception as e:
            error_message = "Molecule {}: {}".format(idx + 1, str(e))
            error_log.append(error_message)
            mol.SetProp("_Error", str(e))
            writer.write(mol)
            error_count += 1

    writer.close()

    if error_file:
        with open(error_file, "w") as ef:
            ef.write("\n".join(error_log))
        print("Error log saved to {}".format(error_file))

    print("Processed {} molecules successfully.".format(processed_count))
    print("Encountered errors with {} molecules.".format(error_count))
    print("Total molecules in input file: {}".format(total_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize chemical structures in an SDF file using RDKit.")
    parser.add_argument("-i", required=True, help="Input SDF file")
    parser.add_argument("-o", required=True, help="Output SDF file")
    parser.add_argument("-e", required=True, help="Error log file")

    args = parser.parse_args()
    standardize_structure(args.i, args.o, args.e)


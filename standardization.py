from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from rdkit import RDLogger
import argparse

RDLogger.DisableLog('rdApp.*')

def copy_properties(original_mol, new_mol):
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            new_mol.SetProp(prop_name, original_mol.GetProp(prop_name))
    return new_mol

# the same list as in check.py
TRANSITION_METALS = [
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    57, 72, 73, 74, 75, 76, 77, 78, 79, 80
]


def has_metal(mol):
    return any(atom.GetAtomicNum() in TRANSITION_METALS for atom in mol.GetAtoms())


def neutralize_atoms(mol):
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    fixes = []
    notes = []
    # molecules read with sanitize=False have no H counts until the cache is computed
    mol.UpdatePropertyCache(strict=False)
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    for at_idx in at_matches_list:
        atom = mol.GetAtomWithIdx(at_idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        old_explicit_hs = atom.GetNumExplicitHs()
        try:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
        except Exception:
            # neutral form is not a valid valence state (e.g. tetra-substituted [B-]), keep the charge
            atom.SetFormalCharge(chg)
            atom.SetNumExplicitHs(old_explicit_hs)
            atom.UpdatePropertyCache(strict=False)
            continue
        sign = "positive" if chg > 0 else "negative"
        notes.append(f"{sign.capitalize()} charge found in {atom.GetSymbol()} atom at position {at_idx + 1}")
        fixes.append(f"Neutralized {sign} charge on {atom.GetSymbol()}")
    if notes:
        mol.SetProp("Neutralization", "\n".join(notes))
    return mol, fixes


def strip_salts(mol):   # I would split the function on two: salt_remover and multi_component_cleaning
                        # for salt removal an rdkit implementations exists https://www.rdkit.org/docs/source/rdkit.Chem.SaltRemover.html
    fixes = []
    original_mol = mol
    salt_smarts = [
        "[H][#8][H]", "Cl[H]", "Br[H]", "I[H]", "F[H]", "[H][#8]-[#7+](-[#8-])=O",
        "[Na+]", "[Ca++]", "[K+]", "[Cl-]", "[Br-]", "[I-]", "Cl", "Br", "I",
        "[#8]S([#8])(=O)=O", "[#8]-[#6](=O)-[#6](-[#8])=O", "[#8-]-[#6](=O)-[#6](-[#8-])=O",
        "[#8]", "[#6]-[#6](-[#8])=O", "[#6]-[#6](-[#8-])=O", "[#8]-[#6](=O)C(F)(F)F",
        "[#8-]-[#6](=O)C(F)(F)F", "[#7]", "[#7+]", "[#6]S([#8])(=O)=O", "[#6]S([#8-])(=O)=O",
        "[#6]-[#6]1=,:[#6]-,:[#6]=,:[#6](-,:[#6]=,:[#6]-,:1)S([#8])(=O)=O",
        "[#6]-[#6]1=,:[#6]-,:[#6]=,:[#6](-,:[#6]=,:[#6]-,:1)S([#8-])(=O)=O",
        "[#8]-[#7+](-[#8-])=O", "[#6]-[#8]S([#8])(=O)=O", "[#6]-[#8]S([#8-])(=O)=O",
        "[#8]-[#6](=O)-[#6]1=,:[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:1",
        "[#8-]-[#6](=O)-[#6]1=,:[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:1",
        "[#6]1=,:[#6]-,:[#6]=,:[#7]-,:[#6]=,:[#6]-,:1", "[#6]1=,:[#6]-,:[#6]=,:[#7+]-,:[#6]=,:[#6]-,:1",
        "[#8]S(=O)(=O)[#6]1=,:[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:1",
        "[#8-]S(=O)(=O)[#6]1=,:[#6]-,:[#6]=,:[#6]-,:[#6]=,:[#6]-,:1",
        "[#6]-[#6]-[#7](-[#6]-[#6])-[#6]-[#6]",
        "[#6]-[#6]-[#7+](-[#6]-[#6])-[#6]-[#6]",
        "[H][#8]S([#6])(=O)=O", "[#8]S(=O)(=O)C(F)(F)F", "O=C=O"
    ]
    if has_metal(mol):
        return mol, []

    salt_mols = [Chem.MolFromSmarts(smarts) for smarts in salt_smarts]
    salts_removed = False  # Track if any salts are removed

    for salt in salt_mols:
        # onlyFrags=True: delete only whole fragments matching the salt, never atoms of a larger molecule
        stripped = Chem.DeleteSubstructs(mol, salt, onlyFrags=True)
        if 0 < stripped.GetNumAtoms() < mol.GetNumAtoms():  # do not delete the last remaining component
            mol = stripped
            salts_removed = True

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        # components which are not in the salt list are dropped here as well
        final_molecule = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
        salts_removed = True
    else:
        final_molecule = Chem.Mol(mol)

    # fragments keep atom order and coordinates, name and properties are restored explicitly
    copy_properties(original_mol, final_molecule)
    if original_mol.HasProp("_Name"):
        final_molecule.SetProp("_Name", original_mol.GetProp("_Name"))

    if salts_removed:
        final_molecule.SetProp("Salt detection", "Presence of salts were detected")
        fixes.append("Largest fragment retained after salt removal")

    try:
        # sanitize in place instead of a SMILES round-trip, which lost name, coordinates and atom order
        Chem.SanitizeMol(final_molecule)
        final_molecule = Chem.RemoveHs(final_molecule)
        Chem.AssignStereochemistry(final_molecule, cleanIt=True, force=True)
    except Exception as e:
        final_molecule.SetProp("Standardization Error", f"Molecule cannot be sanitized: {str(e).strip()}")

    return final_molecule, fixes


def process_aromaticity(mol, report_fix=True):
    # report_fix=False: aromatic flags were set by our own sanitization, converting them back is not a fix
    fixes = []
    if mol is None:
        return mol, ["Molecule is None before aromaticity processing."]

    try:
        if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
                mol.SetProp("Aromaticity Error after salt removal", "Dearomatization failed: Aromatic bonds remain.")
            elif report_fix:
                fixes.append("Dearomatized aromatic system after salt removal.")
        return mol, fixes

    except Exception as e:
        mol.SetProp("Aromaticity Error after salt removal", f"Unexpected error during dearomatization: {e}")
        return mol, fixes

def main():
    parser = argparse.ArgumentParser(description="Process SDF files to strip salts, neutralize charges, and clean 2D structures.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input SDF file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output SDF file.")
    args = parser.parse_args()
    molecule = Chem.SDMolSupplier(args.input, sanitize=False)
    writer = SDWriter(args.output)
    for idx, mol in enumerate(molecule):
        if mol is None or mol.GetNumAtoms() == 0:
            print(f"Warning: Molecule at index {idx} is invalid or empty.")
            placeholder = Chem.Mol()
            placeholder.SetProp("Error", "Invalid or empty molecule")
            writer.write(placeholder)
            continue

        input_aromatic = any(bond.GetIsAromatic() for bond in mol.GetBonds())

        if has_metal(mol):
            # metal-containing molecules are left as they are: no salt removal, no neutralization
            salt_fixes, neutral_fixes = [], []
        else:
            mol, salt_fixes = strip_salts(mol)
            # neutralize after salt removal, so counter-ions do not affect the result
            mol, neutral_fixes = neutralize_atoms(mol)

        mol, aromatic_fixes = process_aromaticity(mol, report_fix=input_aromatic)

        fixes = salt_fixes + neutral_fixes + aromatic_fixes
        if mol.HasProp("Fixes applied") and fixes:
            existing_fixes = mol.GetProp("Fixes applied")
            new_fixes = f"{existing_fixes}\n" + "\n".join(fixes)
            mol.SetProp("Fixes applied", new_fixes)
        elif fixes:
            mol.SetProp("Fixes applied", "\n".join(fixes))
        writer.write(mol)

    writer.close()
    print(f"Processed SDF file saved to: {args.output}")


if __name__ == "__main__":
    main()


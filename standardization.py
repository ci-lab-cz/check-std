from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from rdkit import RDLogger
import argparse

RDLogger.DisableLog('rdApp.*')

def copy_properties(original_mol, new_mol):
    """
    Copies all properties from the original molecule to the new molecule.
    """
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            new_mol.SetProp(prop_name, original_mol.GetProp(prop_name))
    return new_mol

def neutralize_molecule(mol):
    fixes = []
    modified = False
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0 and atom.GetDegree() == 1:  # Check for terminal heteroatom
            symbol = atom.GetSymbol()
            if symbol in ["O", "N", "S", "P"]:  # Common heteroatoms
                atom.SetFormalCharge(0)
                if charge > 0:
                    mol.SetProp("Neutralization",f"Positive charge found in {symbol} atom at position {atom.GetIdx()+1}")
                    atom.SetFormalCharge(0) 
                    fixes.append(f"Neutralized positive charge on terminal {symbol}")
                elif charge < 0:
                    mol.SetProp("Neutralization",f"Negative charge found in {symbol} atom at position {atom.GetIdx()+1}")                    
                    atom.SetFormalCharge(0)
                    fixes.append(f"Neutralized negative charge on terminal {symbol}")
                modified = True
    return mol, fixes

def strip_salts(mol):
    fixes = []
    # SMARTS patterns for salts
    salt_smarts = [
        "[H][#8][H]", "Cl[H]", "Br[H]", "I[H]", "F[H]", "[H][#8]-[#7+](-[#8-])=O",
        "[Na+]", "[Ca++]", "[K+]", "[Cl-]", "[Br-]", "[I-]", "Cl", "Br", "I",
        "[#8]S([#8])(=O)=O", "[#8]-[#6](=O)-[#6](-[#8])=O", "[#8-]-[#6](=O)-[#6](-[#8-])=O",
        "[#8]", "[#6]-[#6](-[#8])=O", "[#6]-[#6](-[#8-])=O", "[#8]-[#6](=O)C(F)(F)F",
        "[#8-]-[#6](=O)C(F)(F)F", "[#7]", "[#7+]", "[#6]S([#8])(=O)=O", "[#6]S([#8-])(=O)=O",
        "[#6]-[#6]-1=[#6]-[#6]=[#6](-[#6]=[#6]-1)S([#8])(=O)=O |c:3,5,t:1|",
        "[#6]-[#6]-1=[#6]-[#6]=[#6](-[#6]=[#6]-1)S([#8-])(=O)=O |c:3,5,t:1|",
        "[#8]-[#7+](-[#8-])=O", "[#6]-[#8]S([#8])(=O)=O", "[#6]-[#8]S([#8-])(=O)=O",
        "[#8]-[#6](=O)-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1 |c:5,7,t:3|",
        "[#8-]-[#6](=O)-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1 |c:5,7,t:3|",
        "[#6]-1=[#6]-[#6]=[#7]-[#6]=[#6]-1 |c:0,2,4|", "[#6]-1=[#6]-[#6]=[#7+]-[#6]=[#6]-1 |c:0,2,4|",
        "[#8]S(=O)(=O)[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1 |c:6,8,t:4|",
        "[#8-]S(=O)(=O)[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1 |c:6,8,t:4|",
        "[#6]-[#6]-[#7](-[#6]-[#6])-[#6]-[#6]",
        "[#6]-[#6]-[#7+](-[#6]-[#6])-[#6]-[#6]",
        "[H][#8]S([#6])(=O)=O", "[#8]S(=O)(=O)C(F)(F)F", "O=C=O"
    ]
    metal = "[Fe,Co,Ni,V,Mn,Cr,Zr,Mo,W,Re,Os,Ru]"
    if mol.HasSubstructMatch(Chem.MolFromSmarts(metal)):
        return mol, []
    else:
        salt_mols = [Chem.MolFromSmarts(smarts) for smarts in salt_smarts if Chem.MolFromSmarts(smarts) is not None]
        salts_removed = False  # Track if any salts are removed

        for salt in salt_mols:
            while mol.HasSubstructMatch(salt):
                match = mol.GetSubstructMatch(salt)
                atom = mol.GetAtomWithIdx(match[0])
                if atom.GetDegree() > 0:
                    break
                else:
                    mol = Chem.DeleteSubstructs(mol, salt)
                salts_removed = True
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)  # Avoid sanitizing fragments prematurely

    # If no fragments remain, return original molecule
    if not fragments:
        return mol, []

    # Retain the largest fragment
    largest_fragment = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms(), default=None)
    if largest_fragment is None or largest_fragment.GetNumHeavyAtoms() < 1:
        return mol, []

    # Convert largest fragment back to molecule and sanitize minimally
    try:
        final_molecule = Chem.MolFromSmiles(Chem.MolToSmiles(largest_fragment, kekuleSmiles=False))
        Chem.AssignStereochemistry(final_molecule, cleanIt=True, force=True)

        # Annotate with `_Error` and `_Fix` fields if salts were removed
        if salts_removed:
            final_molecule.SetProp("Salt detection", "Presence of salts were detected")
            fixes.append("Largest fragment retained after salt removal")

        return final_molecule, fixes
    except Exception as e:
        return mol, []

def process_aromaticity(mol):
    fixes = []
    if mol is None:
        return mol, ["Molecule is None before aromaticity processing."]

    try:
        if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
            mol.SetProp("Aromaticity Error after salt removal", "Molecule contains aromatic bonds.")
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if not any(bond.GetIsAromatic() for bond in mol.GetBonds()):
                fixes.append("Dearomatized aromatic system after salt removal.")
            else:
                mol.SetProp("Aromaticity Error after salt removal", "Dearomatization failed: Aromatic bonds remain.")
        return mol, fixes

    except Exception as e:
        if mol is not None:
            mol.SetProp("Aromaticity Error after salt removal", f"Unexpected error during dearomatization: {e}")
        return mol, fixes

def main():
    parser = argparse.ArgumentParser(description="Process SDF files to strip salts, neutralize charges, and clean 2D structures.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input SDF file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output SDF file.")
    args = parser.parse_args()
    molecule = Chem.SDMolSupplier(args.input, sanitize=False)
    writer = SDWriter(args.output)
    skipped_indices = []
    for idx, mol in enumerate(molecule):
        if mol is None or mol.GetNumAtoms() == 0:
            print(f"Warning: Molecule at index {idx} is invalid or empty.")
            skipped_indices.append(idx)
            placeholder = Chem.Mol()
            placeholder.SetProp("Error", "Invalid or empty molecule")
            writer.write(placeholder)
            continue
        original_mol = mol

        # Neutralize charges
        mol, neutral_fixes = neutralize_molecule(mol)
        mol = copy_properties(original_mol, mol)

        # Strip salts
        mol, salt_fixes = strip_salts(mol)
        mol = copy_properties(original_mol, mol)

        # Process aromaticity
        mol, aromatic_fixes = process_aromaticity(mol)
        mol = copy_properties(original_mol, mol)

        fixes = neutral_fixes + salt_fixes + aromatic_fixes
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


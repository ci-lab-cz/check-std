from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SDWriter
from multiprocessing import Pool
import argparse

RDLogger.DisableLog('rdApp.*')

def copy_properties(original_mol, new_mol):
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            new_mol.SetProp(prop_name, original_mol.GetProp(prop_name))
    return new_mol

def neutralize_molecule(mol):
    fixes = []
    errors = []
    modified = False
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0:  
            symbol = atom.GetSymbol()
            if symbol in ["O", "N", "S", "P"]:  
                atom.SetFormalCharge(0)
                if charge > 0:
                    errors.append(f"Positive charge found in {symbol} atom at position {atom.GetIdx()+1}")
                    atom.SetFormalCharge(0) 
                    fixes.append(f"Neutralized positive charge on atom {symbol}")
                elif charge < 0:
                    errors.append(f"Negative charge found in {symbol} atom at position {atom.GetIdx()+1}")
                    atom.SetFormalCharge(0)
                    fixes.append(f"Neutralized negative charge on atom {symbol}")
                modified = True
        mol.SetProp("Charges found","\n". join(errors))
    return mol, fixes

def strip_salts(mol):
    fixes = []
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
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)  

    if not fragments:
        return mol, []

    largest_fragment = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms(), default=None)
    if largest_fragment is None or largest_fragment.GetNumHeavyAtoms() < 1:
        return mol, []

    try:
        final_molecule = Chem.MolFromSmiles(Chem.MolToSmiles(largest_fragment, kekuleSmiles=False))
        Chem.AssignStereochemistry(final_molecule, cleanIt=True, force=True)

        if salts_removed:
            final_molecule.SetProp("Salt detection", "Presence of salts were detected")
            fixes.append("Largest fragment retained after salt removal")

        return final_molecule, fixes
    except Exception as e:
        return mol, []


def process_mol(mol):
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            return None, []
        original_mol = mol
        original_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown'
        mol, neutral_fixes = neutralize_molecule(mol)
        mol, salt_fixes = strip_salts(mol)
        mol = copy_properties(original_mol, mol)
        mol.SetProp('_Name', original_name)
        fixes = neutral_fixes + salt_fixes
        if mol.HasProp("Fixes applied") and fixes:
            existing_fixes = mol.GetProp("Fixes applied")
            new_fixes = f"{existing_fixes}\n" + "\n".join(fixes)
            mol.SetProp("Fixes applied", new_fixes)
        elif fixes:
            mol.SetProp("Fixes applied", "\n".join(fixes))

        return mol,fixes

    except Exception as e:
        return e, []  # return an exception


def main():
    parser = argparse.ArgumentParser(description="Process SDF files with various functionalities.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file")
    parser.add_argument("-c", "--ncpu", required=False, type=int, default=1, help="CPU count")
    args = parser.parse_args()

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    supplier = Chem.SDMolSupplier(args.input, sanitize=False)
    writer = Chem.SDWriter(args.output)
    skipped_indices = []
    placeholder = Chem.Mol()
    placeholder.SetProp("Error", "Invalid or empty molecule")

    pool = Pool(args.ncpu)
    for idx, (mol, fixes) in enumerate(pool.imap(process_mol, supplier), 1):
        if mol is None:
            print(f"Warning: Molecule at index {idx} is invalid or empty.")
            skipped_indices.append(idx)
            writer.write(placeholder)
        elif isinstance(mol, Exception):
            print(f"Error processing molecule at index {idx}: {mol}")
            skipped_indices.append(idx)
        else:
            writer.write(mol)

    writer.close()
    print(f"Processing complete. Output written to {args.output}")
    if skipped_indices:
        print(f"Skipped molecules at indices: {skipped_indices}")


if __name__ == "__main__":
    main()

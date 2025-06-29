from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SDWriter, rdmolops
import argparse
import logging

RDLogger.DisableLog('rdApp.*')

def copy_properties(original_mol, new_mol):
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            try:
                prop_value = original_mol.GetProp(prop_name)
                new_mol.SetProp(prop_name, prop_value)
            except Exception as e:
                logging.warning(f"Failed to copy property {prop_name}: {e}")
    return new_mol

def neutralize_molecule(mol):
    def is_metal(atom):
        return atom.GetAtomicNum() in {
            3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,
            31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,
            57,72,73,74,75,76,77,78,79,80,81,82,83,84
        }

    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        processed_frags = []
        modified = False

        for frag in fragments:
            rw_frag = Chem.RWMol(frag)
            frag_changed = False

            for atom in rw_frag.GetAtoms():
                charge = atom.GetFormalCharge()
                if charge == 0 or is_metal(atom):
                    continue

                symbol = atom.GetSymbol()
                if symbol not in ['N', 'O', 'S', 'P']:
                    continue

                try:
                    hcount = atom.GetTotalNumHs()
                    if charge < 0:
                        atom.SetFormalCharge(0)
                        atom.SetNumExplicitHs(hcount + abs(charge))
                        frag_changed = True
                    elif charge > 0 and hcount >= charge:
                        atom.SetFormalCharge(0)
                        atom.SetNumExplicitHs(hcount - charge)
                        frag_changed = True
                except:
                    continue

            try:
                rw_frag.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(rw_frag)
                processed_frags.append(rw_frag.GetMol())
                if frag_changed:
                    modified = True
            except:
                processed_frags.append(frag)

        if not processed_frags:
            return mol, []

        emol = Chem.RWMol()
        for frag in processed_frags:
            idx_map = {}
            offset = emol.GetNumAtoms()
            for atom in frag.GetAtoms():
                new_idx = emol.AddAtom(atom)
                idx_map[atom.GetIdx()] = new_idx
            for bond in frag.GetBonds():
                a1 = idx_map[bond.GetBeginAtomIdx()]
                a2 = idx_map[bond.GetEndAtomIdx()]
                emol.AddBond(a1, a2, bond.GetBondType())

        try:
            final = emol.GetMol()
            final.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(final, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            copy_properties(mol, final)
            return final, ["Neutralized"] if modified else []
        except Exception as e:
            return mol, [f"Neutralization fallback: {e}"]

    except Exception as e:
        return mol, [f"Top-level neutralization failure: {e}"]

def strip_salts(mol):
    fixes = []

    # Known isolated salt fragments (as SMARTS)
    salt_smarts = [
        "[Cl-]", "[Br-]", "[I-]", "[Na+]", "[K+]", "[Li+]", "[F-]",
        "[N+](=O)[O-]", "[O-]C=O", "[O-][N+](=O)O"
    ]
    salt_mols = [Chem.MolFromSmarts(s) for s in salt_smarts if s]

    # Split molecule into fragments
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    if not fragments:
        return mol, []

    # Identify and remove isolated salt fragments
    clean_frags = []
    salts_removed = False

    for frag in fragments:
        is_salt = False
        for salt in salt_mols:
            if frag.HasSubstructMatch(salt) and frag.GetNumAtoms() == salt.GetNumAtoms():
                is_salt = True
                break
        if is_salt:
            salts_removed = True
        else:
            clean_frags.append(frag)

    if not clean_frags:
        return mol, []

    # Retain the largest remaining fragment
    largest_fragment = max(clean_frags, key=lambda f: f.GetNumHeavyAtoms())

    final_mol = Chem.RWMol(largest_fragment)
    final_mol.UpdatePropertyCache(strict=False)
    final_mol = final_mol.GetMol()
    copy_properties(mol, final_mol)

    if salts_removed:
        final_mol.SetProp("Salt detection", "Presence of isolated salts were detected")
        fixes.append("Removed isolated salts")
    if len(fragments) > 1:
        fixes.append("Retained largest connected fragment")

    return final_mol, fixes

def process_mol(mol):
    
    try:

        if mol is None or mol.GetNumAtoms() == 0:
            return None, ["Input molecule is None"]

        if any(mol.HasProp(k) for k in ["Kekulization error", "Valence Error", "Processing Error"]):
                return None, [f"Molecule has kekulization/valence/processing error"]

        original_mol = mol
        original_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown'

        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception as e:
            return None, [f"Pre-sanitization failed: {e}"]

        mol, salt_fixes = strip_salts(mol)
        mol, neutralize_fixes = neutralize_molecule(mol)

        charge = Chem.GetFormalCharge(mol)
        if mol.HasProp("Charge"):
            if charge == 0:
                mol.ClearProp("Charge")
            else:
                mol.SetProp("Charge", str(charge))
        elif charge != 0:
            mol.SetProp("Charge", str(charge))

        ncomp = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
        if mol.HasProp("Number of components"):
            if ncomp == 1:
                mol.ClearProp("Number of components")
            else:
                mol.SetProp("Number of components", str(ncomp))
        elif ncomp > 1:
            mol.SetProp("Number of components", str(ncomp))

        fixes = salt_fixes + neutralize_fixes

        if fixes:
            existing = mol.GetProp("Fixes applied") if mol.HasProp("Fixes applied") else ""
            combined = existing.strip().split("\n") if existing else []
            combined += fixes
            mol.SetProp("Fixes applied", "\n".join(filter(None, combined)))
        
        mol.SetProp('_Name', original_name)
        return mol, fixes
    except Exception as e:
        return None, [f"unhandled error during processing: {e}"]

    

def main():
    parser = argparse.ArgumentParser(description="Process SDF files with standardization.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file")
    args = parser.parse_args()

    logging.basicConfig(
        filename='standardization.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    supplier = Chem.SDMolSupplier(args.input, sanitize=False)
    molecules = list(supplier)
    writer = Chem.SDWriter(args.output)
    writer.SetKekulize(False)

    skipped_indices = []

    for idx, mol_in in enumerate(molecules, 1):
        mol, fixes = process_mol(mol_in)
        if mol is None:
            name = mol_in.GetProp("_Name") if mol_in and mol_in.HasProp("_Name") else f"Index {idx}"
            logging.warning(f"Skipped molecule {name}: {fixes}")
            skipped_indices.append(idx)
        else:
            try:
                writer.write(mol)
            except Exception as e:
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Index {idx}"
                logging.error(f"Write failed for molecule {mol_name}: {e}")
                skipped_indices.append(idx)

    writer.close()
    print(f"Processing complete. Output written to {args.output}")
    if skipped_indices:
        print(f"Total molecules skipped: {len(skipped_indices)}")

if __name__ == "__main__":
    main()

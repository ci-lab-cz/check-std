import argparse
from rdkit import Chem
from multiprocessing import Pool
from rdkit.Chem import AllChem, rdmolops, rdDepictor, rdChemReactions, rdMolDescriptors
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ALKALI_METALS = [3, 11, 19, 37, 55, 87]  
ALKALINE_EARTH_METALS = [4, 12, 20, 38, 56, 88]  
HETEROATOMS = [7, 8, 9, 15, 16, 17, 35, 53]  
TRANSITION_METALS = [ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80]
metals = ALKALI_METALS + ALKALINE_EARTH_METALS + TRANSITION_METALS

def copy_properties(original_mol, new_mol):
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            new_mol.SetProp(prop_name, original_mol.GetProp(prop_name))
    return new_mol

def process_molecule(mol):
    errors = []
    valency_issues = []
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        errors.append( f"Invalid molecule block")

    else:
        problem_list = Chem.DetectChemistryProblems(mol)
        if problem_list:
            for problem in problem_list:
                errors.append(f"{problem.GetType()}: {problem.Message()}")

    if errors:
        mol.SetProp("Initial check errors","\n".join(errors))

    return mol

def process_aromaticity(mol):
    fixes = []
    try:
        if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
            mol.SetProp("Aromaticity detection", "Molecule contains aromatic bonds.")
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if not any(bond.GetIsAromatic() for bond in mol.GetBonds()):
                fixes.append("Dearomatized aromatic system.")
            else:
                mol.SetProp("Aromaticity Error", "Dearomatization failed: Aromatic bonds remain.")

        return mol,fixes

    except Exception as e:
        mol.SetProp("Aromaticity Error", f"Unexpected error during dearomatization: {e}")
        return mol,fixes

def clear_incorrectwedge(mol):
    errors = []
    fixes = []
    modified = False

    for bond in mol.GetBonds():
        if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            stereo_valid = (
                begin_atom.HasProp('_CIPCode') or
                end_atom.HasProp('_CIPCode') or
                begin_atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            )

            if not stereo_valid:
                bond.SetBondDir(Chem.BondDir.NONE)
                fixes.append(f"Cleared wedge/dash bond: Bond {bond.GetIdx()}")
                modified = True
            else:
                errors.append(f"Invalid stereochemistry for bond: {bond.GetIdx()}")

    if errors:
        mol.SetProp("Incorrect Wedges", "; ".join(errors))
    
    return mol,fixes


def process_nonstereowedge(mol):
    
    try:
        conf = mol.GetConformer()
        Chem.WedgeMolBonds(mol, conf)
        chiral_centers = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
        fixes = []
        modified = False
        for bond in mol.GetBonds():
            if bond.GetStereo() in (Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW):
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                if begin_atom.GetDegree() < 3 or end_atom.GetDegree() < 3:  # Less than 3 substituents
                    bond.SetStereo(Chem.BondStereo.STEREONONE)
        for bond in mol.GetBonds():
            if bond.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
                start_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                is_valid = (start_atom.GetIdx() in chiral_centers or end_atom.GetIdx() in chiral_centers)

                if not is_valid:
                    start_atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    end_atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    bond.SetBondDir(Chem.BondDir.NONE)  
                    bond.SetStereo(Chem.BondStereo.STEREONONE) 
                    fixes.append(f"Cleared non-stereo wedge bond: Bond {bond.GetIdx()}")
                    modified = True

        if modified:
            mol.SetProp("Non-stereo wedge error", "Non-stereo wedge bonds detected")
    except Exception as e:
        mol.SetProp("_Error", f"Processing error: {str(e)}")

    return mol,fixes

def process_nonstandardwedge(mol):
    fixes = []
    try:
        AllChem.AssignStereochemistry(mol, force=True, cleanIt=True)  # Ensure stereo is assigned first
    
        non_standard = False
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                stereo = bond.GetStereo()
                if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                    continue
                if bond.HasProp("_MolBondDir"):
                    bond_dir = bond.GetProp("_MolBondDir")
                    if bond_dir in ["BEGINWEDGE", "BEGINDASH"]:
                        atom1 = bond.GetBeginAtom()
                        atom2 = bond.GetEndAtom()
                        if atom1.GetChiralTag() not in [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
                                                     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW]:
                            non_standard = True

        if non_standard:
            mol.SetProp("_Error", "Non-standard wedge bond detected")
            mol = copy.deepcopy(mol)
            rdDepictor.Compute2DCoords(mol)  # Ensure 2D coordinates are recomputed
            AllChem.AssignStereochemistry(mol, force=True, cleanIt=True)
            fixes.append(f"Non-Standard wedges have been adjusted")

    except Exception as e:
        mol.SetProp("_Error", f"Processing error: {str(e)}")

    return mol,fixes

def process_metallocene(mol):
    
    fixes = []
    metal_symbols = ["Fe", "Co", "Ni", "Cr", "Mn", "Ru", "V", "Ti"]
    mol = Chem.RWMol(mol)
    metal_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_symbols:
            metal_atoms.append(atom)

    for metal in metal_atoms:
        metal.SetFormalCharge(0)  
        rings = mol.GetRingInfo().AtomRings()
        cyclopentadienyl_rings = [ring for ring in rings if len(ring) == 5 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)]

        if len(cyclopentadienyl_rings) < 2:
            continue

        mol.SetProp("Metallocene detected",f"Metallocene atoms are detected and are not properly bonded with other atoms")
        ring1, ring2 = cyclopentadienyl_rings[:2]
        for ring in [ring1, ring2]:
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetFormalCharge() != 0:  
                    atom.SetFormalCharge(0)  
                    atom.SetNumExplicitHs(0)  

        centroid1 = np.mean([mol.GetConformer().GetAtomPosition(i) for i in ring1], axis=0)
        centroid2 = np.mean([mol.GetConformer().GetAtomPosition(i) for i in ring2], axis=0)
        dummy1_idx = mol.AddAtom(Chem.Atom(0))  
        dummy2_idx = mol.AddAtom(Chem.Atom(0))  
        mol.GetConformer().SetAtomPosition(dummy1_idx, tuple(centroid1))
        mol.GetConformer().SetAtomPosition(dummy2_idx, tuple(centroid2))
        new_position = (centroid1 + centroid2) / 2
        mol.GetConformer().SetAtomPosition(metal.GetIdx(), tuple(new_position))
        mol.AddBond(dummy1_idx, metal.GetIdx(), Chem.BondType.DATIVE)
        mol.AddBond(dummy2_idx, metal.GetIdx(), Chem.BondType.DATIVE)
        fixes.append("Dative bonds are added")
    return mol,fixes

def process_metal(mol):

    has_metal = False  
    mol = Chem.RWMol(mol)  
    errors = []
    fixes = []

    bonds_to_remove = []
    bonds_to_replace = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        a1_num = atom1.GetAtomicNum()
        a2_num = atom2.GetAtomicNum()

        if a1_num in metals or a2_num in metals:
            has_metal = True
            if ((a1_num in ALKALI_METALS + ALKALINE_EARTH_METALS and a2_num in HETEROATOMS) or
                (a2_num in ALKALI_METALS + ALKALINE_EARTH_METALS and a1_num in HETEROATOMS)):
                bonds_to_remove.append((atom1.GetIdx(), atom2.GetIdx()))
                errors.append(f"Bonds detected between metal atom {atom1.GetIdx()+1 if a1_num in ALKALI_METALS + ALKALINE_EARTH_METALS else atom2.GetIdx()+1} and atom {atom2.GetIdx()+1 if a2_num in HETEROATOMS else atom1.GetIdx()+1}.")
            if ((a1_num in TRANSITION_METALS and (a2_num in HETEROATOMS or a2_num == 6)) or
                (a2_num in TRANSITION_METALS and (a1_num in HETEROATOMS or a1_num == 6))):
                bonds_to_replace.append((atom1.GetIdx(), atom2.GetIdx()))
                errors.append(f"Bonds detected between transitional metal atom {atom1.GetIdx()+1 if a1_num in TRANSITION_METALS else atom2.GetIdx()+1} and atom {atom2.GetIdx()+1 if a2_num in HETEROATOMS or a2_num == 6 else atom1.GetIdx()+1}.")

    for idx1, idx2 in bonds_to_remove:
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)
        mol.RemoveBond(idx1, idx2)
        if atom1.GetAtomicNum() in ALKALI_METALS + ALKALINE_EARTH_METALS:
            atom1.SetFormalCharge(atom1.GetFormalCharge() + 1)
            atom2.SetFormalCharge(atom2.GetFormalCharge() - 1)
        elif atom2.GetAtomicNum() in ALKALI_METALS + ALKALINE_EARTH_METALS:
            atom2.SetFormalCharge(atom2.GetFormalCharge() + 1)
            atom1.SetFormalCharge(atom1.GetFormalCharge() - 1)
        fixes.append(f"Bonds between alkali/alkaline earth metals and heteroatoms were broken, and charges were adjusted.")
   
    for idx1, idx2 in bonds_to_replace:
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)
        mol.RemoveBond(idx1, idx2)
        mol.AddBond(idx1, idx2, Chem.BondType.DATIVE)
        Chem.SanitizeMol(mol)
        fixes.append(f"Added dative bond: Atom {idx1+1}  → Atom {idx2+1}.")
        
    if errors:
        mol.SetProp("Metal detection", "\n".join(errors))

    return mol.GetMol(), fixes

def process_transformations(mol):
    transformations = [
    # ("Transform Aromatic N-Oxide", "[#7;a:1]=[O:2]>>[#7+:1]-[#8-:2]"),
     ("Transform Azide", "[#7;X2-:1][N;X2+:2]#[N;X1:3]>>[#7+0:1]=[N+:2]=[#7-:3]"),
     ("Transform Diazo", "[#6;X3-:1][N;X2+:2]#[N;X1:3]>>[#6+0;A:1]=[N+:2]=[#7-:3]"),
     ("Transform Diazonium", "[#6:3]-[#7:1]=[#7+:2]>>[#6:3][N+:1]#[N+0:2]"),
     ("Transform Azo Oxide", "[#8-:3][N:2]#[N+:1]>>[#8-:3]-[#7+:2]=[#7+0:1]"),
     ("Transform Diazo 2", "[#7-:3]=[N:2]#[N+:1]>>[#7+0:1]=[N+:2]=[#7-:3]"),
     ("Transform Iminium", "[#6;X3+:1]-[#7;X3:2]>>[#6+0;A:1]=[#7+:2]"),
     ("Transform Isocyanate", "[#7+:1][#6;A-:2]=[O:3]>>[#7+0:1]=[C+0:2]=[O:3]"),
     ("Transform Nitrilium", "[#6;A;X2+:1]=[#7;X2:2]>>[C+0:1]#[N+:2]"),
     ("Transform Nitro", "[O:3]=[N:1]=[O:2]>>[#8-:2]-[#7+:1]=[O:3]"),
     ("Transform Nitrone Nitronate", "[#6;A:4]=[N:1]=[O:2]>>[#6;A:4]=[#7+:1]-[#8-:2]"),
     ("Transform Nitroso", "[#6:4]-[#7H2+:1]-[#8;X1-:2]>>[#6:4]-[#7+0:1]=[O+0:2]"),
     ("Transform Phosphonic", "[#6][P+:1]([#8;X2:3])([#8;X2:4])[#8-:2]>>[#6][P+0:1]([#8:3])([#8:4])=[O+0:2]"),
     ("Transform Phosphonium Ylide", "[#6][P-:1]([#6])([#6])[#6+:2]>>[#6][P+0:1]([#6])([#6])=[#6+0;A:2]"),
     ("Transform Selenite", "[#8;X2:4][Se+:1]([#8;X2:5])[#8-:2]>>[#8:4][Se+0:1]([#8:5])=[O+0:2]"),
     ("Transform Silicate", "[#8;X2:4]-[#14+:1](-[#8;X2:5])-[#8-:2]>>[#8:4]-[#14+0:1](-[#8:5])=[O+0:2]"),
     ("Transform Sulfine", "[#6]-[#6](-[#6])=[S+:1][#8-:2]>>[#6]-[#6](-[#6])=[S+0:1]=[O+0:2]"),
     ("Transform Sulfon", "[#6][S;X3+:1]([#6])[#8-:2]>>[#6][S+0:1]([#6])=[O+0:2]"),
     ("Transform Sulfonium Ylide", "[#6][S-:1]([#6])[#6+:2]>>[#6][S+0:1]([#6])=[#6+0;A:2]"),
     ("Transform Sulfoxide", "[#6][S+:1]([#6])([#8-:2])=O>>[#6][S+0:1]([#6])(=[O+0:2])=O"),
     ("Transform Sulfoxonium Ylide", "[#6][S+:1]([#6])([#8-:2])=[#6;A]>>[#6][S+0:1]([#6])(=[#6;A])=[O+0:2]"),
     ("Transform Tertiary N-Oxide", "[#6]-[#7;X4:1]=[O:2]>>[#6]-[#7+:1]-[#8-:2]")
    ]
    
    reactions = [(name, rdChemReactions.ReactionFromSmarts(smarts)) for name, smarts in transformations]
    fixes = []
    errors = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    transformed_frags_smiles = []

    for frag in fragments:
        frag.UpdatePropertyCache(strict=False)
        transformed_mols = []
        frag_fixes = []

        for reaction_name, reaction in reactions:
            try:
                products = reaction.RunReactants((frag,))
                unique_smiles = set()
                unique_products = []

                for product_tuple in products:
                    for product in product_tuple:
                        smiles = Chem.MolToSmiles(product)
                        if smiles not in unique_smiles:
                            unique_smiles.add(smiles)
                            unique_products.append(product)

                for product in unique_products:
                    try:
                        Chem.SanitizeMol(product)
                        Chem.rdDepictor.Compute2DCoords(product)
                        errors.append(f"{reaction_name.split(' ', 1)[1]} pattern detected")
                        frag_fixes.append(f"{reaction_name.split(' ', 1)[1]} transformation was successfully applied")
                        transformed_mols.append(product)
                    except Exception:
                        pass

            except Exception:
                pass

        if not transformed_mols:
            transformed_mols.append(frag)

        transformed_frags_smiles.extend([Chem.MolToSmiles(m) for m in transformed_mols])
        fixes.extend(frag_fixes)

    combined_smiles = ".".join(transformed_frags_smiles)
    transformed_mol = Chem.MolFromSmiles(combined_smiles)
    if errors:
        transformed_mol.SetProp("Pattern detection","\n".join(errors))
    return transformed_mol, fixes


def process_undefined_double_bonds(mol):

    def check_nei_bonds(bond):
        Chem.SanitizeMol(mol)
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        a1_bonds_single = [b.GetBondType() == Chem.BondType.SINGLE for b in a1.GetBonds() if b.GetIdx() != bond.GetIdx()]
        a2_bonds_single = [b.GetBondType() == Chem.BondType.SINGLE for b in a2.GetBonds() if b.GetIdx() != bond.GetIdx()]

        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        a1_nei = [a.GetIdx() for a in a1.GetNeighbors() if a.GetIdx() != a2.GetIdx()]
        if len(a1_nei) == 2 and \
                all(mol.GetBondBetweenAtoms(i, a1.GetIdx()).GetBondType() == Chem.BondType.SINGLE for i in a1_nei) and \
                ranks[a1_nei[0]] == ranks[a1_nei[1]]:
            return False
        a2_nei = [a.GetIdx() for a in a2.GetNeighbors() if a.GetIdx() != a1.GetIdx()]
        if len(a2_nei) == 2 and \
                all(mol.GetBondBetweenAtoms(i, a2.GetIdx()).GetBondType() == Chem.BondType.SINGLE for i in a2_nei) and \
                ranks[a2_nei[0]] == ranks[a2_nei[1]]:
            return False

        if a1_bonds_single and a2_bonds_single and \
                all(a1_bonds_single) and all(a2_bonds_single):
            return True
        else:
            return False

    res = []
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE and \
           b.GetStereo() == Chem.BondStereo.STEREONONE and \
           (not b.IsInRing() or not (b.IsInRingSize(3) or b.IsInRingSize(4) or b.IsInRingSize(5) or b.IsInRingSize(6) or b.IsInRingSize(7))) and \
           check_nei_bonds(b):
            res.append(b.GetIdx())
    return res

def process_mol(mol):
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            return None, []
        original_mol = mol
        original_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown'
        mol = process_molecule(mol)
        mol, aromatic_fix = process_aromaticity(mol)
        mol, metallocene_fix = process_metallocene(mol)
        mol, metal_fix = process_metal(mol)
        mol, incorrect_wedge_fix = clear_incorrectwedge(mol)
        mol, non_standard_wedge_fix = process_nonstandardwedge(mol)
        mol, non_stereo_wedge_fix = process_nonstereowedge(mol)
        mol, transformation_fix = process_transformations(mol)
        mol.SetProp('_Name', original_name)
        double = len(process_undefined_double_bonds(mol))
        if double:
            mol.SetProp("Double bonds", str(double))
        tetra = sum(i[1] == '?' for i in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        if tetra:
            mol.SetProp("Undefined stereocenters", str(tetra))
        charge = Chem.GetFormalCharge(mol)
        if charge:
            mol.SetProp("Charge", str(charge))
        ncomp = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
        if ncomp > 1:
            mol.SetProp("Number of components", str(ncomp))
        mol = copy_properties(original_mol, mol)
        fixes = aromatic_fix + metallocene_fix + metal_fix + incorrect_wedge_fix + non_standard_wedge_fix + non_stereo_wedge_fix + transformation_fix
        if fixes:
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



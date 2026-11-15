#!/usr/bin/env python3

import argparse
import logging,re
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import inchi, AllChem, rdmolops, rdDepictor, rdChemReactions, rdMolDescriptors
from rdkit import RDLogger
import numpy as np
import copy
from functools import reduce
import tempfile
patt = re.compile('> {1,2}<(.*)>( +\([0-9]+\))?')



RDLogger.DisableLog('rdApp.*')
def combine_mols_safely(mols):
    if not mols:
        return None
    if len(mols) == 1:
        return mols[0]
    try:
        combined = reduce(Chem.CombineMols, mols)
        return Chem.Mol(combined)
    except Exception as e:
        print(f"[ERROR] combine_mols_safely failed: {e}")
        return None

ALKALI_METALS = [3, 11, 19, 37, 55, 87]
ALKALINE_EARTH_METALS = [4, 12, 20, 38, 56, 88]
HETEROATOMS = [7, 8, 9, 15, 16, 17, 35, 53]
TRANSITION_METALS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80]
metals = ALKALI_METALS + ALKALINE_EARTH_METALS + TRANSITION_METALS

def copy_properties(original_mol, new_mol):
    if original_mol and new_mol:
        for prop_name in original_mol.GetPropNames():
            try:
                prop_value = original_mol.GetProp(prop_name)
                new_mol.SetProp(prop_name, prop_value)
            except Exception as e:
                logging.warning(f"Failed to copy property {prop_name}: {e}")
    return new_mol

def strip_fields(molstr):
    out = []
    i = 0
    while i < len(molstr):
        out.append(molstr[i])
        if patt.fullmatch(molstr[i]):
            tmp = []
            while not patt.fullmatch(molstr[i + 1]) and molstr[i + 1] != '$$$$':
                tmp.append(molstr[i + 1])
                i += 1
            out.extend([line for i, line in enumerate(tmp) if line != '' or i == len(tmp) - 1])
        i += 1
    return out

def process_molecule(mol):
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        msg = "Invalid or empty molecule"
        if mol is None:
            mol = Chem.Mol()
        mol.SetProp("Initial check error", msg)
        return mol
           
    problem_list = Chem.DetectChemistryProblems(mol)
    if problem_list:
        error_msgs = [f"{p.GetType()}: {p.Message()}" for p in problem_list]
        full_error = "\n".join(error_msgs).lower()

        if "valence" in full_error:
            mol.SetProp("Valence error", "\n".join(error_msgs))
            return mol
        elif "kekulize" in full_error or "kekulization" in full_error:
            mol.SetProp("Kekulization error", "\n".join(error_msgs))
            return mol
        else:
            mol.SetProp("Initial check error", "\n".join(error_msgs))
            return mol

    return mol

def process_aromaticity(mol):
    try:
        if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if any(bond.GetIsAromatic() for bond in mol.GetBonds()):
                lem_id = mol.GetProp('LEM') if mol.HasProp('LEM') else 'Unknown'
                logging.warning(f"[LEM: {lem_id}] Dearomatization failed: Aromatic bonds remain.")
                mol.SetProp("Aromaticity errors", "Dearomatization failed: Aromatic bonds remain.")
    except Exception as e:
        lem_id = mol.GetProp('LEM') if mol.HasProp('LEM') else 'Unknown'
        logging.error(f"[LEM: {lem_id}] Kekulization failed: {str(e)}")
        mol.SetProp("Aromaticity errors", f"Kekulization failed: {str(e)}")

    return mol

def is_nitrogen_bridgehead(mol, atom):
    
    if atom.GetAtomicNum() != 7: 
        return False
    if atom.GetDegree() < 3:  
        return False
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.NumAtomRings(atom.GetIdx())
    return atom_rings >= 2

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
                begin_atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED or
                is_nitrogen_bridgehead(mol, begin_atom) or  
                is_nitrogen_bridgehead(mol, end_atom)       
            )

            if not stereo_valid:
                bond.SetBondDir(Chem.BondDir.NONE)
                fixes.append(f"Cleared wedge/dash bond: Bond {bond.GetIdx()}")
                modified = True
            else:
                errors.append(f"Invalid stereochemistry for bond: {bond.GetIdx()}")

    if errors:
        mol.SetProp("Incorrect Wedges", "; ".join(errors))
    
    return mol, fixes

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

                is_valid = (
                    start_atom.GetIdx() in chiral_centers or 
                    end_atom.GetIdx() in chiral_centers or
                    is_nitrogen_bridgehead(mol, start_atom) or  
                    is_nitrogen_bridgehead(mol, end_atom)       
                )

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

    return mol, fixes

def process_nonstandardwedge(mol):
    fixes = []
    try:
        AllChem.AssignStereochemistry(mol, force=True, cleanIt=True)  
    
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
                        if is_nitrogen_bridgehead(mol, atom1) or is_nitrogen_bridgehead(mol, atom2):
                            continue
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

    return mol, fixes

def process_metallocene(mol):

    metal_symbols = ["Fe", "Co", "Ni", "Cr", "Mn", "Ru", "V", "Ti"]
    mol = Chem.RWMol(mol)
    metal_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_symbols:
            metal_atoms.append(atom)

    for metal in metal_atoms:
        if metal.GetDegree() == 0:
            continue
        metal.SetFormalCharge(0)
        rings = mol.GetRingInfo().AtomRings()
        cyclopentadienyl_rings = [
            ring for ring in rings
            if len(ring) == 5 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)
        ]

        if len(cyclopentadienyl_rings) < 2:
            continue

        mol.SetProp("Metallocene detected", "Metallocene atoms are detected and are not properly bonded with other atoms")

        ring1, ring2 = cyclopentadienyl_rings[:2]
        for ring in [ring1, ring2]:
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetFormalCharge() != 0:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)

    return mol

def process_metal(mol):
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
        fixes.append("Bonds between alkali/alkaline earth metals and heteroatoms were broken, and charges were adjusted.")

        if errors:
            mol.SetProp("Metal detection", "\n".join(errors))

    return mol.GetMol(), fixes

def process_transformations(mol):
    transformations = [
        ("Transform Aromatic N-Oxide", "[#7;a:1]=[O:2]>>[#7+:1]-[#8-:2]"),
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
        ("Transform Tertiary N-Oxide", "[#6]-[#7;X4:1]=[O:2]>>[#6]-[#7+:1]-[#8-:2]"),
    ]

    reactions = [(name, rdChemReactions.ReactionFromSmarts(smarts)) for name, smarts in transformations]
    fixes = []
    matched_patterns = []
    any_transformation_applied = False

    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception as e:
        return None, [f"GetMolFrags failed: {e}"]

    final_mols = []

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
                        try:
                            Chem.SanitizeMol(product)
                            smi = Chem.MolToSmiles(product)
                            if smi not in unique_smiles:
                                unique_smiles.add(smi)
                                unique_products.append(product)
                        except:
                            continue

                if unique_products:
                    any_transformation_applied = True
                    matched_patterns.append(reaction_name.replace("Transform ", "") + " pattern detected")
                    pattern_name = reaction_name.replace("Transform ", "")
                    frag_fixes.append(f"{pattern_name} transformation applied")

                    transformed_mols.extend(unique_products)
            except Exception:
                continue

        if not transformed_mols:
            transformed_mols.append(frag)

        final_mols.extend(transformed_mols)
        fixes.extend(frag_fixes)
    if not any_transformation_applied:
        return mol, []

    transformed_mol = combine_mols_safely(final_mols)
    if transformed_mol is None:
            return None, ["CombineMols failed after transformation"]
    else:
            transformed_mol.SetProp("Pattern detection", ", ".join(matched_patterns))
    return transformed_mol, fixes
def process_unspec_double_bonds(mol):
    def check_nei_bonds(bond):
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
            return None, ["initial molecule is None or empty"]

        original_mol = mol
        mol = process_molecule(mol)
        for error_key in ["Initial check error", "Valence error", "Kekulization error"]:
            if mol.HasProp(error_key):
                error_msg = mol.GetProp(error_key)
                original_mol.SetProp(error_key, error_msg)

                # Set _Name from LEM if available
                if original_mol.HasProp('LEM'):
                    original_mol.SetProp('_Name', original_mol.GetProp('LEM'))

                return original_mol, [f"{error_key}: {error_msg}"]



        mol = process_aromaticity(mol)
        aromatic_errors = mol.GetProp("Aromaticity errors") if mol.HasProp("Aromaticity errors") else None
        if aromatic_errors:
            original_mol.SetProp("Processing error", aromatic_errors)
            return original_mol, [f"Aromaticity error: {aromatic_errors}"]

        try:
            double = len(process_unspec_double_bonds(mol))
            if double:
                mol.SetProp("Double bonds", str(double))
        except Exception as e:
            return None, [f"process_unspec_double_bonds failed: {e}"]
        
        mol = process_metallocene(mol)
        if mol is None:
            return None

        mol, metal_fix = process_metal(mol)
        if mol is None:
            return None, ["process_metal returned None"]

        mol, incorrect_wedge_fix = clear_incorrectwedge(mol)
        if mol is None:
            return None, ["clear_incorrectwedge returned None"]

        mol, non_standard_wedge_fix = process_nonstandardwedge(mol)
        if mol is None:
            return None, ["process_nonstandardwedge returned None"]

        mol, non_stereo_wedge_fix = process_nonstereowedge(mol)
        if mol is None:
            return None, ["process_nonstereowedge returned None"]

        mol, transformation_fix = process_transformations(mol)
        if mol is None:
            return None, ["process_transformations returned None"]

        try:
            lem_id = mol.GetProp("LEM") if mol.HasProp("LEM") else None
            mol.SetProp('_Name', lem_id)
        except Exception:
            pass

       

        try:
            Chem.SanitizeMol(mol)
            AllChem.AssignStereochemistry(mol, force=True, cleanIt=True)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

            tetra = sum(i[1] == '?' for i in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            if tetra:    
                mol.SetProp("Undefined stereocenters", str(tetra))
        except Exception as e:
            return None, [f"FindMolChiralCenters failed: {e}"]

        try:
            charge = Chem.GetFormalCharge(mol)
            if charge:
                mol.SetProp("Charge", str(charge))
        except Exception as e:
            return None, [f"GetFormalCharge failed: {e}"]

        try:
            ncomp = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
            if ncomp > 1:
                mol.SetProp("Number of components", str(ncomp))
        except Exception as e:
            return None, [f"GetMolFrags failed: {e}"]
        
        try:
            mol = copy_properties(original_mol, mol)
        except Exception as e:
            return None, [f"copy_properties failed: {e}"]

        fixes = ( metal_fix +
            incorrect_wedge_fix + non_standard_wedge_fix +
            non_stereo_wedge_fix + transformation_fix
        )

        if fixes:
            try:
                mol.SetProp("Fixes applied", "\n".join(fixes))
            except Exception:
                pass

        return mol, fixes

    except Exception as e:
        return None, [f"process_mol wrapper failed: {e}"]

def process_with_lem(args):
    idx, mol, lem_id = args
    try:
        result, fixes_or_errors = process_mol(mol)
        return idx, result, fixes_or_errors, lem_id
    except Exception as e:
        return idx, None, [f"process_with_lem failed: {e}"], lem_id

def write_molecule(writer, mol, preserve_properties=True):
    """Write molecule to SDWriter while preserving all properties exactly"""
    if preserve_properties:
        # Get all properties before writing
        props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
        
        # Write the molecule
        writer.write(mol)
        
        # After writing, restore all properties exactly as they were
        for k, v in props.items():
            mol.SetProp(k, v)
    else:
        writer.write(mol)

def check(input_file, output_file, ncpu, logfile):
    logging.basicConfig(
        filename=logfile,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(input_file) as infile, tempfile.NamedTemporaryFile(mode='w+', delete=False) as cleaned:
        molstr = []
        for line in infile:
            molstr.append(line.rstrip())
            if line.strip() == '$$$$':
                molstr = strip_fields(molstr)
                cleaned.write('\n'.join(molstr) + '\n')
                molstr = []
        cleaned_filename = cleaned.name
    supplier = Chem.SDMolSupplier(cleaned_filename, sanitize=False)
    writer = Chem.SDWriter(output_file)

    inputs = []
    for idx, mol in enumerate(supplier, 1):
        if mol is None:
            continue

        lem_id = mol.GetProp("LEM") if mol.HasProp("LEM") else None
        inputs.append((idx, mol, lem_id))

    with Pool(ncpu) as pool:
        for idx, result, fixes_or_errors, lem_id in pool.imap(process_with_lem, inputs):
            orig_mol = inputs[idx - 1][1]

            error_keys = ["Initial check error", "Valence error", "Kekulization error"]
            has_specific_error = result is None or any(result.HasProp(k) for k in error_keys)

            if has_specific_error:
                reason = "; ".join(fixes_or_errors) if fixes_or_errors else "Unknown reason"
                logging.warning(f"Error in molecule {lem_id} at index {idx}. Reason: {reason}")

                if orig_mol is not None:
                    if orig_mol.HasProp('LEM'):
                        orig_mol.SetProp('_Name', orig_mol.GetProp('LEM'))

                    if result is not None:
                        for key in error_keys:
                            if result.HasProp(key):
                                orig_mol.SetProp(key, result.GetProp(key))
                                break
                        else:
                            orig_mol.SetProp("Processing error", reason)
                    else:
                        orig_mol.SetProp("Processing error", reason)

                    try:
                        writer.write(orig_mol)
                    except Exception as e:
                        logging.warning(f"Write failed for molecule {lem_id} at index {idx}: {e}")
                        try:
                            writer.SetKekulize(False)
                            writer.write(orig_mol)
                        except Exception as e2:
                            logging.error(f"Final fallback failed for {lem_id} at index {idx}: {e2}")
                continue

            try:
                Chem.SanitizeMol(result)
            except Exception as e:
                result.SetProp("Sanitization error", str(e))

            if not result.HasProp('_Name') and result.HasProp('LEM'):
                result.SetProp('_Name', result.GetProp('LEM'))

            for prop_name in orig_mol.GetPropNames():
                try:
                    prop_value = orig_mol.GetProp(prop_name)
                    result.SetProp(prop_name, prop_value)
                except Exception as e:
                    logging.warning(f"Failed to copy property {prop_name} on molecule {lem_id}: {e}")

            try:
                write_molecule(writer, result)
            except Exception as e:
                logging.warning(f"Write failed for molecule {lem_id} at index {idx}: {e}")
                try:
                    result.setKekulize(False)
                    write_molecule(writer, result)
                except Exception as e2:
                    logging.error(f"Second write attempt failed for {lem_id} at index {idx}: {e2}")

    writer.close()
    logging.info(f"Processing complete. Output written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Check and fix molecules while preserving all properties exactly.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file")
    parser.add_argument("-c", "--ncpu", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument("-l", "--logfile", required=True, help="Log file path")

    args = parser.parse_args()
    check(
        input_file=args.input,
        output_file=args.output,
        ncpu=args.ncpu,
        logfile=args.logfile
    )

if __name__ == "__main__":
    main()
import argparse
from rdkit import Chem
from rdkit.Chem import rdAbbreviations


def extract_metadata(input_sdf):

    with open(input_sdf, "r") as infile:
        content = infile.read()

    mol_blocks = content.split("$$$$")
    metadata = []
    for mol_block in mol_blocks:
        mol_block = mol_block.strip()
        if mol_block:
            lines = mol_block.split("\n")
            mol_metadata = {
                "raw_block": mol_block,
                "header": lines[0],  
                "fields": {},  
                "block_lines": [],
            }
            in_field = False
            field_name = None
            for line in lines[1:]:
                if line.startswith(">  <") and "> " in line:
                    in_field = True
                    field_name = line.split("<")[1].split(">")[0]
                    mol_metadata["fields"][field_name] = []
                elif in_field:
                    if line.strip() == "":
                        in_field = False
                    else:
                        mol_metadata["fields"][field_name].append(line.strip())
                else:
                    mol_metadata["block_lines"].append(line)

            for field, values in mol_metadata["fields"].items():
                mol_metadata["fields"][field] = "\n".join(values)
            metadata.append(mol_metadata)
    return metadata


def process_pseudo_atoms(mol, abbrev_dict):
    pseudo_atom_errors = []
    if mol is not None:
        try:
            editable_mol = Chem.RWMol(mol)
            for atom in mol.GetAtoms():
                if atom.HasProp("_Name"):
                    label = atom.GetProp("_Name")
                    if label in abbrev_dict:
                        fragment = abbrev_dict[label]
                        combined = Chem.CombineMols(editable_mol, fragment)
                        combined_editable = Chem.RWMol(combined)
                        atom_idx = atom.GetIdx()
                        frag_start_idx = mol.GetNumAtoms()
                        combined_editable.AddBond(atom_idx, frag_start_idx, Chem.BondType.SINGLE)
                        editable_mol = combined_editable
                    else:
                        pseudo_atom_errors.append(f"Unreplaceable pseudo-atom: {label}")
            mol = editable_mol.GetMol()
        except Exception as e:
            pseudo_atom_errors.append(f"Error during pseudo-atom processing: {str(e)}")

    return mol, pseudo_atom_errors


def rebuild_sdf(metadata, processed_molecules, output_sdf, error_sdf):
    valid_blocks = []
    error_blocks = []

    for meta, (mol, pseudo_atom_errors) in zip(metadata, processed_molecules):
        new_fields = meta["fields"].copy()
        new_fields["PseudoAtomErrors"] = "; ".join(pseudo_atom_errors) if pseudo_atom_errors else "None"

        block_lines = [meta["header"]] + meta["block_lines"]
        for field, value in new_fields.items():
            block_lines.append(f">  <{field}>")
            block_lines.append(value)
            block_lines.append("")
        block_lines.append("$$$$")

        if pseudo_atom_errors:
            error_blocks.append("\n".join(block_lines))
        else:
            valid_blocks.append("\n".join(block_lines))

    with open(output_sdf, "w") as valid_file:
        valid_file.write("\n".join(valid_blocks))
    with open(error_sdf, "w") as error_file:
        error_file.write("\n".join(error_blocks))


def process_sdf(input_sdf, output_sdf, error_sdf):
    abbrevs = rdAbbreviations.GetDefaultAbbreviations()
    abbrev_dict = {abbr.label: abbr.mol for abbr in abbrevs}

    metadata = extract_metadata(input_sdf)
    supplier = Chem.SDMolSupplier(input_sdf)
    if not supplier:
        raise ValueError("Failed to read the input SDF file.")

    processed_molecules = []
    for meta, mol in zip(metadata, supplier):
        if mol is None:
            processed_molecules.append((None, ["Invalid molecule in input SDF"]))
            continue
        mol, pseudo_atom_errors = process_pseudo_atoms(mol, abbrev_dict)
        processed_molecules.append((mol, pseudo_atom_errors))

    rebuild_sdf(metadata, processed_molecules, output_sdf, error_sdf)

    print(f"Valid molecules saved to: {output_sdf}")
    print(f"Molecules with pseudo-atom errors saved to: {error_sdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SDF for pseudo-atom replacement.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    args = parser.parse_args()

    process_sdf(args.input, args.output, args.error)


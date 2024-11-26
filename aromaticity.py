import argparse
from rdkit import Chem


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
                "fields": {},
                "block_lines": [],
            }
            in_field = False
            field_name = None
            for line in lines:
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


def rebuild_sdf(molecules, output_filename):

    with open(output_filename, "w") as outfile:
        for mol, meta in molecules:
            if mol is None:
                # Handle invalid molecules gracefully
                outfile.write(f"{meta['raw_block']}\n")
                outfile.write(">  <AromaticityError>\n")
                outfile.write("Error: Invalid molecule block\n")
                outfile.write("$$$$\n")
                continue

            # Add metadata fields
            mol_block = Chem.MolToMolBlock(mol)
            outfile.write(mol_block)
            outfile.write("\n")

            for field, value in meta["fields"].items():
                outfile.write(f">  <{field}>\n")
                outfile.write(f"{value}\n\n")

            if mol.HasProp("AromaticityError"):
                outfile.write(">  <AromaticityError>\n")
                outfile.write(f"{mol.GetProp('AromaticityError')}\n\n")

            outfile.write("$$$$\n")


def process_and_dearomatize(metadata, input_sdf):

    supplier = Chem.SDMolSupplier(input_sdf)
    fixed_mols = []
    error_mols = []

    for meta, mol in zip(metadata, supplier):
        error_message = None

        if mol is None:
            error_message = "Error: Invalid molecule block."
        else:
            try:
                # Attempt to dearomatize the molecule
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
                dearomatized_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                dearomatized_mol = Chem.MolFromSmiles(dearomatized_smiles)
                if dearomatized_mol:
                    mol = dearomatized_mol
                    Chem.SanitizeMol(mol)
                else:
                    error_message = "Error: Failed to dearomatize molecule."
            except Chem.KekulizeException:
                error_message = "Error: Can't Kekulize molecule."
            except Exception as e:
                error_message = f"Error during processing: {str(e)}"


        if error_message:
            if mol:
                mol.SetProp("AromaticityError", error_message)
            error_mols.append((mol, meta))
        else:
            if mol:
                mol.SetProp("AromaticityError", "None")
            fixed_mols.append((mol, meta))

    return fixed_mols, error_mols


def main():
    parser = argparse.ArgumentParser(description="Process SDF files for aromaticity.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file for valid molecules")
    parser.add_argument("-e", "--error", required=True, help="Output SDF file for error molecules")
    args = parser.parse_args()

    metadata = extract_metadata(args.input)
    fixed_mols, error_mols = process_and_dearomatize(metadata, args.input)
    rebuild_sdf(fixed_mols, args.output)
    rebuild_sdf(error_mols, args.error)

if __name__ == "__main__":
    main()


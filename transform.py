import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

def apply_transformations(input_file, output_file, error_file):
    transformations = {
    "Aromatic N-Oxide": "[N:1]=[O:2]>>[N+:1]-[O-:2]",
    #"Azide": "[N:1][N:2]=[N:3]>>[N:1]=[N+:2]=[N-:3]",
    #"Diazo": "[C:1][N:2]=[N:3]>>[C:1]=[N+:2]=[N-:3]",
    "Diazonium": "[C:1]-[N:2]=[N+:3]>>[C:1][N+:2]#[N:3]",
    "Azo Oxide": "[O-:1][N:2]=[N+:3]>>[O-:1]-[N+:2]=[N:3]",
    "Diazo 2": "[N-:1]=[N:2]=[N+:3]>>[N:1]=[N+:2]=[N-:3]",
    #"Iminium": "[C:1]-[N:2]>>[C:1]=[N+:2]",
    "Isocyanate": "[N+:1][C:2]=[O:3]>>[N:1]=[C:2]=[O:3]",
    "Nitrilium": "[C+:1]=[N:2]>>[C:1]#[N+:2]",
    "Nitro": "[O:1]=[N:2]=[O:3]>>[O-:3]-[N+:2]=[O:1]",
    "Nitrone Nitronate": "[C:1]=[N:2]=[O:3]>>[O-:3]-[N+:2]=[C:1]",
    "Nitroso": "[C:1]-[N:2]-[O:3]>>[C:1]-[N:2]=[O:3]",
    "Phosphonic": "[P+:1]([O:2])([O:3])[O-:4]>>[P:1]([O:2])([O:3])=[O:4]",
    "Phosphonium Ylide": "[P-:1]([C:2])([C:3])[C+:4]>>[P:1]([C:2])([C:3])=[C:4]",
    "Selenite": "[O:1][Se+:2][O:3][O-:4]>>[O:1][Se:2]([O:3])=[O:4]",
    "Sulfine": "[C:1]-[S+:2][O-:3]>>[C:1]=[S:2]=[O:3]",
    "Sulfoxide": "[C:1][S+:2][O-:3]>>[C:1][S:2]=[O:3]",
    "Tertiary N-Oxide": "[N:1]=[O:2]>>[N+:1]-[O-:2]"
}

    suppl = Chem.SDMolSupplier(input_file)
    writer = Chem.SDWriter(output_file)
    error_writer = Chem.SDWriter(error_file)

    for idx, mol in enumerate(suppl):
        if mol is None:
            error_message = "Could not parse molecule from input file."
            print(f"Molecule {idx+1}: {error_message}")
            error_mol = Chem.Mol()
            error_mol.SetProp("_Name", f"Molecule {idx+1}")
            error_mol.SetProp("Error", error_message)
            error_writer.write(error_mol)
            continue

        error_message = "None"
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)

            for name, reaction_smarts in transformations.items():
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
                    substructure = Chem.MolFromSmarts(reaction_smarts.split(">>")[0])
                    if not mol.HasSubstructMatch(substructure):
                        continue

                    products = rxn.RunReactants((mol,))
                    if products:
                        product = products[0][0]
                        Chem.SanitizeMol(product)
                        #print(f"Valid product after '{name}': {Chem.MolToSmiles(product)}")
                        mol = product
                except Exception as e:
                    error_message = f"Error in reaction '{name}': {e}"
                    raise e

            mol.SetProp("Error", error_message)
            writer.write(mol)

        except Exception as e:
            print(f"Error processing Molecule {idx+1}: {e}")
            mol.SetProp("Error", error_message)
            error_writer.write(mol)

    writer.close()
    error_writer.close()
    print(f"Transformation complete. Successfully processed molecules saved to {output_file}")
    print(f"Failed molecules saved to {error_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply molecular transformations to an SDF file.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-o", "--output", required=True, help="Output SDF file (successful molecules)")
    parser.add_argument("-e", "--error", required=True, help="Error log file (failed molecules)")

    args = parser.parse_args()
    apply_transformations(args.input, args.output, args.error)

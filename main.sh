#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Define the input file
input_file=$1

# Define the names of the Python scripts
scripts=("check_chem.py" "pseudo.py" "aromaticity.py" "ionic.py" "bond_check.py" "metal.py" "transform.py" "final_standardization.py")

# Directory to store intermediate files
output_dir="outputs"
error_dir="errors"

# Create directories if they do not exist
mkdir -p "$output_dir"
mkdir -p "$error_dir"

# File to combine all error molecules
combined_error_file="$error_dir/combined_errors.sdf"
> "$combined_error_file" # Clear or create the file

# Initialize the current input
current_input=$input_file

# Iterate over the scripts and chain them
for script in "${scripts[@]}"; do
    # Define output and error files for this script
    output_file="$output_dir/${script%.py}_output.sdf"
    error_file="$error_dir/${script%.py}_errors.sdf"

    # Execute the Python script
    echo "Running $script with input: $current_input"
    python "$script" -i "$current_input" -o "$output_file" -e "$error_file"

    # Check if script execution was successful
    if [ $? -ne 0 ]; then
        echo "Error running $script. Check $error_file for details."
        exit 1
    fi

    # Append the error molecules to the combined error file
    if [ -s "$error_file" ]; then
        cat "$error_file" >> "$combined_error_file"
    fi

    # Update the current input to the output file
    current_input=$output_file
done

echo "Processing complete."
echo "Final output: $current_input"
echo "Combined error file: $combined_error_file"


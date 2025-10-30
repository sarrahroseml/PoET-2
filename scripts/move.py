import gemmi
import os

def list_files_with_extension(directory_path, extension):
    """
    Lists all files in a given directory that end with the specified extension.
    """
    files_with_extension = []
    for filename in os.listdir(directory_path):
        if filename.endswith(extension) and os.path.isfile(os.path.join(directory_path, filename)):
            files_with_extension.append(filename)
    return files_with_extension
my_list = list_files_with_extension("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/pdb", ".cif")

# Replace 'your_structure.cif' with the actual path to your CIF file
# Replace 'output_structure.pdb' with the desired path for the output PDB file
for file in my_list:
    name = file.split(".")[0]
    try:
        # Read the CIF file
        structure = gemmi.read_structure(file)

        # Write the structure to a PDB file
        structure.write_minimal_pdb(name + ".pdb")

        print(f"Successfully converted '{file}' to '{name}.pdb'.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        print(name)
   
import os
import shutil
import argparse

def copy_files_with_filter(source_folder, look_for):
    # Define the destination folder as 'output' inside the source folder
    destination_folder = os.path.join(os.path.dirname(source_folder), f"{os.path.basename(source_folder)}_{look_for}_filtered")
    os.makedirs(destination_folder, exist_ok=False)

    # Iterate over all files in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if look_for in file:
                name, ext = os.path.splitext(file)
                new_name = name.replace('.', '_') + ext
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)

                # Copy the file to the destination folder
                shutil.copy(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Copy files containing a specific string in their name to a '<source>_filtered' folder next to the source folder."
    )
    parser.add_argument("--folder", help="Path to the source folder")
    parser.add_argument("--look-for", default="CH1", help="String to look for in file names (default: 'CH1')")

    args = parser.parse_args()

    copy_files_with_filter(args.folder, args.look_for)

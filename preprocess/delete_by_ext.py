from pathlib import Path

# Usage example
folder_path = "/gpfs/mariana/home/darobn/datasets/madasr23dataset/bn.cleaned"
extension = ".spec.pt"
log_frequency = 2000  # Log info every 1000 removed files


def remove_files_with_extension(folder_path, extension):
    p = Path(folder_path)
    files = list(
        p.rglob(f"*{extension}")
    )  # Use rglob method to search for files recursively

    counter = 0  # Counter for removed files

    for file in files:
        file.unlink()  # Use Path.unlink() for more efficient file removal
        counter += 1

        if counter % log_frequency == 0:
            print(f"Removed {counter} files")

    print(f"Total removed files: {counter}")


remove_files_with_extension(folder_path, extension)

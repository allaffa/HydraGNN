import os
import subprocess
import traceback
from .file_utils import contains_subdirs


def extract_tar_file(tar_file, dest, create_subdir=True):
    """
    Extract a tar_file at the destination location represented by 'dest'.
    If dest is None, use the cwd as dest.
    if create_subdir is True, create a subdirectory inside dest in which the tar file will be extracted.

    Returns the dir path in which the tar file was extracted along with a list of molecule directories in
    the tar file.
    """

    tarfname = os.path.basename(tar_file).split(".tar")[0]

    # Set the destination directory based on input args
    if dest is not None and create_subdir is True:
        dest = os.path.join(os.path.abspath(dest), tarfname)
    if dest is None:
        dest = os.path.join(os.getcwd(), tarfname)

    # Create the dest directory
    os.makedirs(dest, exist_ok=True)

    # Launch the tar command to extract the tar file
    untar_cmd = "tar -xf {} -C {}".format(tar_file, dest)
    p = subprocess.run(untar_cmd.split())
    p.check_returncode()

    return dest


def get_mol_dir_list(tar_path):
    """Get a list of molecule directories from the extracted tar

    The tar may contain a list of molecule directories or a high-level directory that contains
    molecule directories.
    """
    mol_dirs = [os.path.join(tar_path, mol_dir) for mol_dir in next(os.walk(tar_path))[1]]

    # Extracted tar may contain a single high-level directories of mol directories
    if len(mol_dirs) == 1 and contains_subdirs(mol_dirs[0]):
        return get_mol_dir_list(mol_dirs[0])

    return mol_dirs

import os


def contains_subdirs(dirpath):
    # Return True if there are subdirectories in dirpath
    dirs = [f for f in os.scandir(dirpath) if f.is_dir()]
    return True if len(dirs) > 0 else False

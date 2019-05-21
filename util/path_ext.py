from os import path, sep
from sys import argv


def replace_ext(file_path: str, new_ext: str):
    return path.splitext(file_path)[0] + new_ext


def add_subfolder_and_replace_ext(file_path: str, subfolder: str, new_ext: str):
    data_dir, basename = path.split(file_path)
    return path.join(data_dir, subfolder, path.splitext(basename)[0] + new_ext)


def path_fix(path_to_fix: str):
    if path.exists(path_to_fix):
        return path_to_fix
    fixed_path = path.join('.', path_to_fix)
    return fixed_path if path.exists(fixed_path) else path_to_fix


def script_folder():
    return path.dirname(path.abspath(argv[0]))

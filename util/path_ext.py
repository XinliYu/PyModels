from os import path
from sys import argv


def path_fix(path_to_fix: str):
    if path.exists(path_to_fix):
        return path_to_fix
    fixed_path = path.join('.', path_to_fix)
    return fixed_path if path.exists(fixed_path) else path_to_fix


def script_folder():
    return path.dirname(path.abspath(argv[0]))

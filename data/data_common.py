from os import path
from datetime import datetime

results_folder_name = 'results'


def output_file_name(data_folder: str, extension:str='.txt'):
    return path.join(data_folder, results_folder_name, datetime.now().strftime('%m%d%Y_%H%M%S') + extension)

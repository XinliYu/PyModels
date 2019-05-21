import urllib.request
from os import path, remove
from util.path_ext import *
from data.data_load import *
from util.io_ext import *
from util.general_ext import *
import zipfile

supported_glove_dimensions = (50, 100, 200, 300)
glove_folder = 'glove'


def glove_download():
    """
    Downloads and unzips GloVe word embeddings.
    """
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_file_name = glove_url.split()[-1]
    print('Downloading glove word embedding file from ' + glove_url)
    dest_path = path.join(path.dirname(path.abspath(__file__)), glove_folder, glove_file_name)
    download(glove_url, dest_path)
    print('Unzipping ' + glove_file_name)
    unzip(dest_path, path.dirname(dest_path))
    print('Deleting ' + glove_file_name)
    remove(dest_path)


def load_glove_embedding(dimension: int):
    """
    Loads GloVe embeddings of a specified dimension. Supported dimensions include 50, 100, 200, 300.
    See https://nlp.stanford.edu/projects/glove/.
    :param dimension: specify the dimension of the embedding.
    :return: a dictionary with the keys as the words, and the values as their embeddings.
    """
    if dimension in supported_glove_dimensions:
        glove_embed_file_name = 'glove.6B.{}d'.format(dimension)
        data_file = path.join(path.dirname(path.abspath(__file__)), glove_folder, glove_embed_file_name)
        binary_data_file = data_file + '.dat'
        if path.exists(binary_data_file):
            return load(data_file=binary_data_file,
                        data_format=SupportedDataFormats.DictTuples,
                        value_type_or_format_fun=float,
                        compressed=True)
        else:
            txt_data_file = data_file + '.txt'
            if not path.exists(txt_data_file):
                glove_download()
            glove_dict = load(data_file=txt_data_file,
                              data_format=SupportedDataFormats.DictTuples,
                              value_type_or_format_fun=float)
            save(binary_data_file, glove_dict, compressed=True)
            remove(txt_data_file)
            return glove_dict
    else:
        error_print(error_tag=path.abspath(__file__), message='glove vectors only available in the following dimensions: ' + str(supported_glove_dimensions))


if __name__ == '__main__':
    print('Testing glove embedding load ...')
    for dim in supported_glove_dimensions:
        print(len(load_glove_embedding(dim)))

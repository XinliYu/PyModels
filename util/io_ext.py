import urllib.request
import gzip
import pickle
import sys
from zipfile import ZipFile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dest_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def unzip(src_path, dest_dir, filter=None):
    with ZipFile(file=src_path) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            # Extract each file to the `dest_dir` if it is in `filter`;
            # if you want to extract to current working directory, don't specify path.
            if filter is None or (isinstance(filter, list) and file in filter) or (isinstance(filter, str) and file == filter):
                zip_file.extract(member=file, path=dest_dir)


def pickle_load(file_path: str, compressed: bool = False, encoding=None):
    with open(file_path, 'rb') if not compressed else gzip.open(file_path, 'rb') as f:
        if encoding is None or sys.version_info < (3, 0):
            return pickle.load(f)
        else:
            return pickle.load(f, encoding=encoding)


def pickle_save(file_path: str, data, compressed: bool = False):
    with open(file_path, 'wb') if not compressed else gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)

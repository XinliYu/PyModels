from typing import List, Tuple
import scipy.sparse as sp
import numpy as np
import pickle as pkl
import codecs


def mask1d_by_index(mask_idxes, size: int, ref_idxes=None):
    if type(mask_idxes[0]) in (list, np.ndarray):
        mask_count = len(mask_idxes)
        masks = [np.zeros(size, dtype=bool) for _ in range(mask_count)]
        if ref_idxes is None:
            for i in range(mask_count):
                masks[i][mask_idxes] = True
        else:
            for j, ref_idx in enumerate(ref_idxes):
                for i in range(mask_count):
                    if ref_idx in mask_idxes[i]:
                        masks[i][j] = True
        return tuple(masks)

    else:
        mask = np.zeros(size, dtype=bool)
        if ref_idxes is None:
            mask[mask_idxes] = True
        else:
            for i, ref_idx in enumerate(ref_idxes):
                if ref_idx in mask_idxes:
                    mask[i] = True
        return mask


def labels_to_onehot(labels):
    min_label = min(labels)
    max_label = max(labels)
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    labels = labels - min_label
    label_count = len(labels)
    one_hots = np.zeros((label_count, max_label - min_label + 1))
    one_hots[np.arange(label_count), labels] = 1
    return one_hots


def sample_index_by_percentage(size: int, percents: Tuple[float]) -> List[np.ndarray]:
    permuted_indices = np.random.permutation(size)
    split_count = len(percents)
    index_samples = [None] * split_count
    start = 0

    for i in range(split_count):
        end_p = percents[i]
        if end_p == 0:
            continue
        end = start + int(end_p * size) if i != split_count - 1 else size
        index_samples[i] = permuted_indices[start:end]
        start = end
    return index_samples


def mask_by_percentage(size: int, percents: Tuple):
    index_samples = sample_index_by_percentage(size, percents)
    mask_count = len(percents)
    masks = [None] * mask_count
    for i in range(mask_count):
        indices = index_samples[i]
        if indices is None:
            continue
        masks[i] = np.zeros(size, dtype=bool)
        masks[i][index_samples[i]] = True

    return masks


def save_lists(file_path: str, lists: List, delimiter=' ', encoding=None):
    list_count = len(lists[0])
    if file_path.endswith('.txt'):
        with open(file_path, 'w') if encoding is None else codecs.open(file_path, 'w', encoding) as f:
            for i in range(list_count):
                f.write(delimiter.join([str(l[i]) for l in lists if l is not None]))
                f.write('\n')
            f.flush()
    else:
        with open(file_path, 'wb') as f:
            pkl.dump(lists, f)


def sparse_to_tuple(sparse_mat):
    if not sp.isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape


def sparse_row_normalize(a):
    row_sum = np.array(a.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    a = r_mat_inv.dot(a)
    return sparse_to_tuple(a)


def row_normalize(a: np.ndarray):
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]

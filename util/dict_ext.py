def index_dict(enumerable):
    idx_dict = {}
    for idx, item in enumerate(enumerable):
        idx_dict[item] = idx
    return idx_dict

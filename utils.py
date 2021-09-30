import copy
import sys


def update_dict(ori, update):
    try:
        d = copy.deepcopy(ori)
        for k, v in update.items():
            if k in d.keys():
                d[k] = v
            else:
                raise KeyError
    except KeyError:
        print('[get_dict] A key does not exist.')
        sys.exit(1)
    return d



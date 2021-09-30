import numpy as np


def svd_np_fast_ccr(W, rank, double_precision=False):

    def solve(W):
        WtW = np.dot(W.T, W)
        V, S2, VT = np.linalg.svd(WtW, full_matrices=False, compute_uv=True)
        S = np.sqrt(S2)
        U = np.dot(np.dot(W, V), np.linalg.inv(np.diag(S)))         # This will throw an error for the singular matrix.
        return U, S, VT

    def svd(W):
        size = W.shape
        m, n = size[0], size[1]
        if m > n:
            return solve(W)
        elif m < n:
            V, S, UT = solve(W.T)
            return UT.T, S, V.T
        else:
            return np.linalg.svd(W, full_matrices=False, compute_uv=True)

    r = np.max([np.min([W.shape[0], W.shape[1], rank]), 1])
    U, S, VT = svd(W.astype(np.float64) if double_precision else W)
    ccr = 0.999999999 if double_precision else 0.999999
    R = np.where(np.cumsum(S * S) / np.sum(S * S) >= ccr)[0]
    if R.size != 0:
        S[R[0] + 1:] = 0.
    if double_precision:
        return U[:, :r].astype(np.float32), S[:r].astype(np.float32), VT[:r, :].astype(np.float32)
    else:
        return U[:, :r], S[:r], VT[:r, :]


# W: a list of numpy 2D matrices
def full_svd_cpu(W, algorithm='numpy_f_ccr'):
    try:
        if algorithm == 'numpy_f_ccr':
            return [svd_np_fast_ccr(w, rank=1000000, double_precision=False) for w in W]
        else:
            print('algorithm: %s is not found' % algorithm)
            return None

    except np.linalg.LinAlgError as err:
        print('numpy.linalg.LinAlgError: ', err)
        return None


def sort_singular_values(SV):
    s = np.hstack([sv for sv in SV])
    l = np.hstack([[k] * sv.shape[0] for k, sv in enumerate(SV)])
    s = np.hstack([np.reshape(l, [-1, 1]), np.reshape(s, [-1, 1])])
    order = np.argsort(s, axis=0)[::-1][:, 1]
    return s[order]


def singular_value_rank(sorted_sv, r_ratio, rank_min=1):
    r = np.max([0., np.min([1.0, r_ratio])])    
    r_min = np.max([1, rank_min])
    num = np.round(r * len(sorted_sv)).astype('int32')
    l = sorted_sv[:, 0].astype('int32')
    lr = l[:num]
    rank = np.histogram(lr, bins=np.arange(np.max(l)+2))[0]
    return np.clip(rank, r_min, None).tolist()

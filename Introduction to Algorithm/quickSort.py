def _partition(A, p, r):
    """
    Params:
        A: {list}
        p: {int} start index
        r: {int} end index
    Notes:
    -   以第一个数为基准，小于其的数放左边，大于其的数放右边
    """
    q = p
    for i in range(p + 1, r):
        if A[q] > A[i]:
            A[q], A[i] = A[i], A[q]
            q = i
    return q

def _qsort(A, p, r):
    """
    Params:
        A: {list}
        p: {int} start index
        r: {int} end index
    Notes:
        sort A[p, r)
    """
    if p == r: return A

    q = _partition(A, p, r)
    _qsort(A, p, q)
    _qsort(A, p + 1, r)
    return A

def qsort(A, inverse=False):
    """
    Params:
        A: {list}
    """
    A = _qsort(A, 0, len(A))
    A = A[::-1] if inverse else A
    return A

if __name__ == "__main__":
    A = list(range(4, 25))[::-1]
    print(A)
    qsort(A)
    print(A)
def _merge_list(l1, l2):
    i = j = 0; l = []
    while True:
        if l1[i] < l2[j]:
            l += [l1[i]]
            i += 1
        else:
            l += [l2[j]]
            j += 1
        
        if i == len(l1):
            l += l2[j:]
            break
        elif j == len(l2):
            l += l1[i:]
            break
    return l

def _merge(A):
    """
    Params:
        A: {list[list]}
    """
    n = len(A)
    T = []
    if n % 2:
        T += [A.pop(-1)]
    for i in range(n//2):
        T += [_merge_list(A[i*2], A[i*2+1])]
    return T

def msort(A, inverse=False):
    """
    Params:
        A: {list}
    """
    T = [[i] for i in A]

    while len(T) > 1:
        T = _merge(T)

    T = T[0]
    T = T[::-1] if inverse else T

    return T

if __name__ == "__main__":
    A = list(range(4, 25))[::-1]
    print(A)
    A = msort(A)
    print(A)
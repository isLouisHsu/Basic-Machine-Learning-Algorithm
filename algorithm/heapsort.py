def _is_max_heapify(A, i):
    n = len(A)
    l = 2 * i + 1; r = l + 1

    l = l if l < n else i
    r = r if r < n else i

    return (A[i] >= A[l]) and (A[i] >= A[r])

def _max_heapify(A, i):
    """
    Params:
        A: {list}
        i: {int}
    Returns:
        A: {list}
    """
    n = len(A)
    if n == 0:
        return A

    l = 2 * i + 1; r = l + 1
    
    largest = l if l < n and A[l] > A[i] else i
    largest = r if r < n and A[r] > A[largest] else largest

    A[largest], A[i] = A[i], A[largest]

    if not _is_max_heapify(A, largest):
        _max_heapify(A, largest)

    return A

def _build_max_heap(A):
    n = len(A) // 2
    for i in range(n)[::-1]:
        A = _max_heapify(A, i)
    return A

def heapsort(A, inverse=False):
    n = len(A)

    ## 先建最大堆
    A = _build_max_heap(A)
    
    H = []
    for i in range(n):
        ## 将最大值A[0]与最后元素交换，并弹出
        A[0], A[-1] = A[-1], A[0]
        H += [A.pop(-1)]
        ## 重新更新最大堆
        A = _max_heapify(A, 0)
    
    return H

if __name__ == "__main__":
    A = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    H = heapsort(A)
    print(H)
def csort(A):
    C = [0 for i in range(max(A))]
    ## 统计各数据的数目
    for a in A:
        C[a-1] += 1
    ## 统计各数据的累计数目
    for i in range(1, len(C)):
        C[i] = C[i] + C[i-1]
    
    T = [0 for i in range(len(A))]
    for a in A:
        ## 查询当前数据在队列中的位置
        T[C[a-1]-1] = a
        ## 当前数据个数位置减1
        C[a-1] -= 1
    return T

if __name__ == "__main__":
    A = list(range(4, 25))[::-1] + list(range(4, 25))[::-1]
    print(A)
    A = csort(A)
    print(A)
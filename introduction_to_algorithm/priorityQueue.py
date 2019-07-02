import numpy as np
from heapSort import _build_max_heap, _max_heapify

def _max_priority_queue(A, i):
    """
    Params:
        A: {list}
        i: {int} index of current number
    """
    while i > 0:
        p = (i - 1) // 2    # 父节点索引
        if A[i] <= A[p]:
            break
        
        A[i], A[p] = A[p], A[i]
        i = p
    return A

class PriorityQueue():
    """
    Attributes:
        nums: {list}
    """
    def __init__(self, nums=[]):
        self.nums = _build_max_heap(nums)

    def __len__(self):
        return len(self.nums)

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return self.extract_maximun()
        except:
            raise StopIteration

    def __repr__(self):
        
        ret = ""

        n = len(self); h = int(np.ceil(np.log2(n)))
        for i in range(h):
            n = (2**(h) - 2**(i+1))
            ret = "{:s}{:s}".format(ret, " "*n)
            s = 2**i - 1; e = s + 2**i
            nums = self.nums[s: e]
            for num in nums:
                ret = "{:s}{:<4d}".format(ret, num)
            ret += "\n"
        return ret


    def insert(self, num):
        """
        Params:
            num: {numerical object}
        """
        self.nums += [num]      # 插入到最后节点位置
        _max_priority_queue(self.nums, len(self)-1) # 上浮
    
    def maximum(self):
        return self.nums[0]
    
    def pop(self):
        ## 最后一个节点交换至最后位置
        self.nums[0], self.nums[-1] = self.nums[-1], self.nums[0]
        maxNum = self.nums.pop(-1)
        _max_heapify(self.nums, 0)  # 第一个节点下降
        return maxNum

    def change(self, i, num):
        """
        Params:
            i: {int} index of origin number
            num: {numerical object}
        """
        if i >= len(self): return

        temp = self.nums[i]
        self.nums[i] = num      # 修改该值
        if num >= temp:         # 上浮
            _max_priority_queue(self.nums, i)
        else:                   # 下降
            _max_heapify(self.nums, i)

if __name__ == "__main__":
    A = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    pq = PriorityQueue(A)
    print(pq)

    ## ========================================
    pq.insert(17)
    print(pq)
    pq.insert(15)
    print(pq)

    ## ========================================
    pq.change(1, 1)
    print(pq)
    pq.change(1, 18)
    print(pq)

    ## ========================================
    pq.pop()
    print(pq)
    pq.pop()
    print(pq)

    ## ========================================
    for num in pq:
        print(num)
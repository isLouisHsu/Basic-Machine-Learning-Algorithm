class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        l, r = self._findMaxSubArray(nums, 0, len(nums))
        return nums[l: r]
    
    def _findMaxSubArray(self, nums, low, high):
        """
        Params:
            nums: {list[int]}
            low:  {int}
            high: {int}
        Return:
            index: {tuple[int]}
        Notes:
            find maximum sub-array        
        """
        if len(nums[low: high]) == 1:
            return (low, high)
        
        mid = (low + high) // 2
        
        indexL = self._findMaxSubArray(nums, low, mid)
        indexR = self._findMaxSubArray(nums, mid, high)
        indexC = self._findMaxCrossingSubArray(nums, low, high)
        indexes = [indexL, indexR, indexC]
        
        sum_ = list(map(lambda x: sum(nums[x[0]: x[1]]), indexes))
        index = indexes[sum_.index(max(sum_))]
        
        return index
        
    def _findMaxCrossingSubArray(self, nums, low, high):
        """
        Params:
            nums: {list[int]}
            low:  {int}
            high: {int}
        Return:
            index: {tuple[int]}
        Notes:
            find maximum crossing sub-array        
        """
        mid = (low + high) // 2
        
        sumL = float('-inf'); idxL = low
        for i in range(low, mid):
            sum_ = sum(nums[i: mid])
            if sum_ > sumL:
                sumL = sum_
                idxL = i
        
        sumR = float('-inf'); idxR = high
        for i in range(mid, high + 1):
            sum_ = sum(nums[mid: i])
            if sum_ > sumR:
                sumR = sum_
                idxR = i
        
        return (idxL, idxR)
            
if __name__ == "__main__":
    A = [-2, 2, -3, 4, -1, 2, 1, -5, 3]
    S = Solution().maxSubArray(A)
    print(S)

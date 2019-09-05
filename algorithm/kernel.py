# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-05 12:42:49
@LastEditTime: 2019-09-05 13:45:13
@Update: 
'''
import numpy as np

class Kernel:

    @staticmethod    
    def gaussian(x, y, sigma):
        """ Gaussian Kernel
        Params: 
            x, y: {ndarray(n_features)}
            sigma: {float}
        Notes:
            K(x, y) = \exp (- || x - y ||^2 / 2 \sigma^2)
        """
        return np.exp(-0.5 * np.linalg.norm(x - y) / sigma**2)

    @staticmethod
    def polynomial(x, y, c, d):
        """ Polyomial Kernel
        Params:
            x, y: {ndarray(n_features,)}
            c, d: {float} constant
        Notes:
            K(x, y) = (x^T y + c)^d, c > 0
        """
        return (np.dot(x, y) + c) ** d
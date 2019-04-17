import numpy as np
from collections import OrderedDict
from sklearn.decomposition import PCA as skPCA
from p50_pca import PCA as myPCA

class NDarrayPCA(object):
    """ PCA for ndarray

    Attributes:
        n_dims:         {int}               number of dimension of input data
        n_components:   {list[int/None]}    number of components of each dimension
        decomposers:    {OrderedDict}       index: PCA
    """
    def __init__(self, n_components=None, use_sklearn=True):

        self.n_dims = None
        self.n_components = [] if n_components is None else n_components
        self.decomposers = OrderedDict()
        self.use_sklearn = use_sklearn

    def fit(self, X):
        """
        Params:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        """

        self.n_dims = len(X.shape) - 1
        idx = [i for i in range(len(X.shape))]   # index of dimensions

        for i_dim in range(self.n_dims):
            self.decomposers[i_dim] = skPCA(n_components=self.__n_components(i_dim)) \
                        if self.use_sklearn else myPCA(n_component=self.__n_components(i_dim))
            
            ## transpose tensor
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
            X = X.transpose(idx)
            shape = list(X.shape)

            # 1-dim pca
            X = X.reshape((-1, shape[-1]))
            X = self.decomposers[i_dim].fit_transform(X)

            ## transpose tensor
            X = X.reshape(shape[:-1]+[X.shape[-1]])
            X = X.transpose(idx)
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]

    def transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        Returns:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        """

        assert self.n_dims == len(X.shape) - 1, 'please check input dimension! '
        idx = [i for i in range(len(X.shape))]   # index of dimensions

        for i_dim in range(self.n_dims):
            
            ## transpose tensor
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
            X = X.transpose(idx)
            shape = list(X.shape)

            # 1-dim pca
            X = X.reshape((-1, shape[-1]))
            X = self.decomposers[i_dim].transform(X)

            ## transpose tensor
            X = X.reshape(shape[:-1]+[X.shape[-1]])
            X = X.transpose(idx)
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
        
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X

    def transform_inv(self, X):
        """
        Params:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        Returns:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        """
        
        if self.use_sklearn:
            raise NotImplementedError

        idx = [i for i in range(len(X.shape))]   # index of dimensions

        for i_dim in range(self.n_dims, 0, -1):

            ## transpose tensor
            idx[-1], idx[i_dim] = idx[i_dim], idx[-1]
            X = X.transpose(idx)
            shape = list(X.shape)

            # 1-dim pca
            X = X.reshape((-1, shape[-1]))
            X = self.decomposers[i_dim-1].transform_inv(X)

            ## transpose tensor
            X = X.reshape(shape[:-1]+[X.shape[-1]])
            X = X.transpose(idx)
            idx[-1], idx[i_dim] = idx[i_dim], idx[-1]
        
        return X
    
    def __n_components(self, index):
        
        try:
            return self.n_components[index]
        except IndexError:
            return None


if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt

    X = []
    for i in range(20):
        X += [cv2.imread("./data/obj{}__0.png".format(i+1))]
    X = np.concatenate(list(map(lambda x: x[np.newaxis], X)), axis=0)
    
    # show first image
    plt.figure(0)
    plt.imshow(X[0, :, :, :])

    decomposer = NDarrayPCA(n_components=[64, 32, None], use_sklearn=False)
    # decomposer = NDarrayPCA(n_components=None, use_sklearn=False)
    decomposer.fit(X)

    X_transformed = decomposer.transform(X)

    # show first decomposed image
    plt.figure(1)
    plt.imshow(X_transformed[0, :, :, :])

    X_transformed_inv = decomposer.transform_inv(X_transformed).astype('uint8')

    # show first inverse-decomposed image
    plt.figure(2)
    plt.imshow(X_transformed_inv[0, :, :, :])
    plt.show()
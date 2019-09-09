import numpy as np

class MinimumDistance():
    """ Minimum Distance Classifier
    Attributes:
        means_: {ndarray(n_classes, n_features)}
    Notes:
        $$ c^{(j)} = \frac{1}{N_j} \sum_{i=1}^{N_j} x^{(i)} $$
        $$ \hat{y}^{(i)} = \arg \min || x^{(i)} - c^{(j)} || $$
    Example:
        ``` python
        clf = MinimumDistance()
        X = np.random.rand(20, 10)
        y = np.array([np.random.randint(0, 2) for _ in range(20)])

        clf.fit(X, y)
        y_ = clf.predict(X)
        ```
    """
    def __init__(self):

        self.means_ = None

    def fit(self, X, y):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
            y: {ndarray(n_samples)}
        """
        labels = list(set(y))
        means_ = []

        for i_classes in range(len(labels)):

            label = labels[i_classes]
            X_ = X[y==label]
            means_ += [np.mean(X_, axis=0)]
        
        self.means_ = np.array(means_)

    def predict(self, X):
        """
        Params:
            X: {ndarray(n_samples, n_features)}
        """
        if len(X.shape) == 1:
            raise ValueError("Please reshape input array as `X.reshape(1, -1)` if number of sample is `1`")
        
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples, dtype=np.int)

        for i in range(X.shape[0]):

            dists = np.linalg.norm(X[i] - self.means_, axis=1)
            y_pred[i] = np.argmin(dists)

        return y_pred

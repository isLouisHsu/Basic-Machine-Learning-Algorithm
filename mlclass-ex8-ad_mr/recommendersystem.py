import numpy as np
from load_data import load_movie_review_R_Y

class CollaborativeFilter():
    '''
    @note: 根据输入的矩阵，学习特征和参数
    '''
    def __init__(self, n_features):
        self.n_features = n_features
        self.n_movies = None
        self.n_users = None
        self.X = None # 电影的特征矩阵 n_movies   × n_features
        self.W = None # 用户的参数矩阵 n_features x n_users
    def fit(self, R, Y, learing_rate=0.01, regularize=0.0, max_iter=5000, min_loss=0.0):
        '''
        @param {bool array} R: an binary-valued indicator matrix
        @param {float array} Y: stores the ratings
        '''
        def gradient():
            Y_pred = self.X.dot(self.W) # n_movies × n_users
            E = Y_pred - Y              # n_movies × n_users
            gradX = (R * E).dot(self.W.T) + regularize * self.X    # n_movies x n_features
            gradW = self.X.T.dot(R * E) + regularize * self.W      # n_features x n_users
            return gradX, gradW
        # initialize parameters
        self.n_movies = R.shape[0]  # or Y.shape[0]
        self.n_users  = R.shape[1]  # or Y.shape[1]
        self.X = np.random.rand(self.n_movies, self.n_features)
        self.W = np.random.rand(self.n_features, self.n_users)
        # Implementational Detail_Mean Normalization
        Y_normalized = Y - np.mean(Y, axis=1).reshape(-1, 1)
        # gradient descent
        n_iter = 0; loss_now = float('inf')
        while n_iter < max_iter:
            n_iter += 1
            gradX, gradW = gradient()
            self.X -= learing_rate * gradX
            self.W -= learing_rate * gradW
            Y_pred = self.X.dot(self.W)
            # todo
        pass
    def predict(self, idx_m, idx_u):
        '''
        @param {list[int]} idx_m, idx_u
        @note: predicts the rating for movie i(s) by user j(s) 
        @example: predict([1, 2, 3], [1, 2]) —— predicts the rating for movie 1, 2, 3 by user 1, 2 
        '''
        return self.X[idx_m, :].dot(W[:, idx_u])
    def score_MSE_regularized(self, Y_true, Y_pred, R, regularize=0.0):
        Err = Y_true - Y_pred
        loss_1 = np.sum((E * R)**2)
        loss_2 = regularize * np.sum(self.X**2)
        loss_3 = regularize * np.sum(self.W**2)
        return 0.5 * (loss_1 + loss_2 + loss_3)

if __name__ == '__main__':
    R, Y = load_movie_review_R_Y()

    estimator = CollaborativeFilter(n_features=10)
    estimator.fit(R, Y)
    
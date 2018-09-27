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
            Y_pred = self.X.dot(self.W) * self.Y_std + self.Y_mean # n_movies × n_users
            E = Y_pred - Y                                         # n_movies × n_users
            gradX = (R * E).dot(self.W.T) + regularize * self.X    # n_movies x n_features
            gradW = self.X.T.dot(R * E) + regularize * self.W      # n_features x n_users
            return gradX, gradW
        # initialize parameters
        self.n_movies = R.shape[0]  # or Y.shape[0]
        self.n_users  = R.shape[1]  # or Y.shape[1]
        self.X = np.random.rand(self.n_movies, self.n_features)
        self.W = np.random.rand(self.n_features, self.n_users)
        # Normalization
        self.Y_mean = np.mean(Y, axis=1).reshape((-1, 1))
        self.Y_std  = np.std(Y, axis=1).reshape((-1, 1))
        Y_normalized = (Y - self.Y_mean) / self.Y_std   # 将每一个用户对某一部电影的评分减去所有 用户对该电影评分的平均值
        # gradient descent
        n_iter = 0; loss_now = float('inf')
        while n_iter < max_iter:
            n_iter += 1
            gradX, gradW = gradient()
            self.X -= learing_rate * gradX
            self.W -= learing_rate * gradW
            Y_pred = self.X.dot(self.W) * self.Y_std + self.Y_mean
            loss_now = self.score_MSE_regularized(Y, Y_pred, R, regularize=0.0)
            if loss_now < min_loss:
                print('收敛，最终迭代%d次，当前损失值为%f' % (n_iter, loss_now))
            if n_iter % 100 == 0:
                print('第%d次迭代，当前损失值为%f' % (n_iter, loss_now))

        if n_iter >= max_iter:
            print('超过迭代次数，当前损失值为%f' % loss_now)

    def predict(self, idx_m, idx_u):
        '''
        @param {list[int]} idx_m, idx_u
        @note: predicts the rating for movie i(s) by user j(s) 
        @example: predict([1, 2, 3], [1, 2]) —— predicts the rating for movie 1, 2, 3 by user 1, 2 
        '''
        return self.X[idx_m, :].dot(self.W[:, idx_u]) * self.Y_std + self.Y_mean
    def score_MSE_regularized(self, Y_true, Y_pred, R, regularize=0.0):
        E = Y_true - Y_pred
        loss_1 = np.sum((E * R)**2)
        loss_2 = regularize * np.sum(self.X**2)
        loss_3 = regularize * np.sum(self.W**2)
        return 0.5 * (loss_1 + loss_2 + loss_3)

if __name__ == '__main__':
    # R, Y = load_movie_review_R_Y()
    R = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 0]
    ]).astype('bool')
    Y = np.array([
        [5, 5, 0, 0],
        [5, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 5, 4],
        [0, 0, 5, 0]
    ])

    estimator = CollaborativeFilter(n_features=50)
    estimator.fit(R, Y)
    Y_pred = estimator.predict(idx_m=[0, 1, 2, 3, 4], idx_u=[0, 1, 2, 3]).astype('int')
    print(Y_pred)
    '''
    n_features = 10
    [[5 5 0 0]
    [5 5 1 0]
    [3 4 0 0]
    [0 0 5 3]
    [0 0 5 3]]

    n_features = 50
    [[5 5 0 0]
    [5 2 0 0]
    [0 4 0 0]
    [0 0 5 4]
    [0 0 5 0]]
    '''
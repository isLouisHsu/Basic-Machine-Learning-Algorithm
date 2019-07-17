import numpy as np
import pandas as pd
import scipy.sparse as sp

def rmse(gt, pred):
    """
    Params:
        gt: {ndarray(n_users, n_items)}
        pred: {ndarray(n_users, n_items)}
    Returns:
        score: {float}
    Notes:
        \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y - \hat{y})^2}
    """
    mask = gt != 0
    err = gt[mask] - pred[mask]
    score = np.sqrt(np.mean(err**2))
    return score

def load(path):
    """
    Params:
        path: {str}
    Returns:
        user_id: {ndarray(n)}
        item_id: {ndarray(n)}
        rating:  {ndarray(n)}
    Notes:
        - 数据来自[MoiveLens](http://files.grouplens.org/datasets/movielens/ml-100k.zip)
    """
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    
    df = pd.read_csv(path, sep='\t', names=header)
    n_users = max(df.user_id)
    n_items = max(df.item_id)
    
    print("Number of users = %d, Number of movies = %d" % (n_users, n_items))
    user_id = df.loc[:, 'user_id'].values
    item_id = df.loc[:, 'item_id'].values
    rating  = df.loc[:, 'rating' ].values

    return user_id, item_id, rating

def get_matrix(raw_data, shape, return_sparse=False):
    """
    Params:
        raw_data: {tuple(3)} user_id, item_id, rating
        shape:    {tuple(2)} rows, cols
        return_sparse: {bool} 
    Returns:
        m: {ndarray, sparse matrix}
    Notes:
        convert raw data to matrix(n_users, n_items)
    """
    user_id, item_id, rating = raw_data
    m = sp.coo_matrix((rating, (user_id-1, item_id-1)), shape)
    if not return_sparse:
        m = m.toarray()
    return m

def get_cosine_similarity_matrix(x):
    """
    Params:
        x: {ndarray(n_users, n_features)}
    """
    x_normd = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    cosine = x_normd.dot(x_normd.T)
    return cosine

def predict(train_matrix, test_matrix, compress=80, n_keep=50):

    # Singular value decomposition
    _, _, vh = np.linalg.svd(train_matrix)

    # Compress matrix
    train_compressed = train_matrix.dot(vh.T[:, : compress])# N x compress

    # Calculate similarity matrix
    similarity = get_cosine_similarity_matrix(train_compressed)
        
    prediction = np.zeros_like(test_matrix)             # to preserve the predicted results
    to_pred = np.array(np.where(test_matrix != 0))      # the indexes to be predicted, shape(2, n)
    
    # predict
    for i in range(to_pred.shape[1]):

        r, c = to_pred[:, i]                            # `r` is the index of user, `c` is the the index of item

        id = np.argsort(similarity[r])[::-1]            # sort samples according to similarity in descending order
        id = id[1: n_keep + 1]                          # top `n_keep` users
        rates = train_matrix[id, c]                     # get the ratings of chosen samples
        rates = rates[rates!=0]                         # filter non-zero data

        rate = np.mean(rates) if rates.shape[0] != 0 else 0
        prediction[r, c] = rate

    return prediction

def according_to_user(train_matrix, test_matrix, compress=80, n_keep=50):
    prediction = predict(train_matrix, test_matrix, compress, n_keep)
    return prediction

def according_to_item(train_matrix, test_matrix, compress=80, n_keep=100):
    prediction = predict(train_matrix.T, test_matrix.T, compress, n_keep).T
    return prediction

if __name__ == "__main__":

    N_USERS, N_ITEMS = 943, 1682
    shape = (N_USERS, N_ITEMS)
    
    ## 载入数据
    train_raw = load('./u1.base')
    train_matrix = get_matrix(train_raw, shape)
    test_raw  = load('./u1.test')
    test_matrix  = get_matrix(test_raw,  shape)

    ## 预测
    pred_according_to_user = according_to_user(train_matrix, test_matrix)   # 根据用户相似度
    pred_according_to_item = according_to_item(train_matrix, test_matrix)   # 根据电影相似度

    # 共同作用：若均有值，取均值；若均无值，取0；否则填充其中一个
    user_nzero = pred_according_to_user != 0
    item_nzero = pred_according_to_item != 0
    nzero = np.bitwise_and(user_nzero, item_nzero)
    
    pred_according_to_user_and_item = pred_according_to_user + pred_according_to_item
    pred_according_to_user_and_item[nzero] = pred_according_to_user_and_item[nzero] / 2

    ## 打印指标
    print("[user] rmse = ", rmse(test_matrix, pred_according_to_user))
    print("[item] rmse = ", rmse(test_matrix, pred_according_to_item))
    print("[total] rmse = ", rmse(test_matrix, pred_according_to_user_and_item))


    ## 作图显示
    from matplotlib import pyplot as plt

    plt.figure(0)
    plt.subplot(211)
    plt.imshow(train_matrix) 
    plt.subplot(212)
    plt.imshow(test_matrix) 

    plt.figure(1)
    plt.subplot(221)
    plt.imshow(test_matrix) 
    plt.subplot(222)
    plt.imshow(pred_according_to_user) 
    plt.subplot(223)
    plt.imshow(pred_according_to_item) 
    plt.subplot(224)
    plt.imshow(pred_according_to_user_and_item)
    plt.show()

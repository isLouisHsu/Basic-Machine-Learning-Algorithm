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

def according_to_user(train_matrix, test_matrix, cols=80, n_keep=50):

    # 将矩阵SVD分解
    _, _, vh = np.linalg.svd(train_matrix)

    # 压缩原矩阵，A' = A V[:, :k]
    train_compressed_col = train_matrix.dot(vh[: cols].T)   # N_USERS x cols

    # 计算相似度矩阵
    similarity_user = get_cosine_similarity_matrix(train_compressed_col)

    # 预测
    pred_matrix = np.zeros_like(test_matrix)        # 保存预测结果
    to_pred = np.array(np.where(test_matrix != 0))  # 需要预测的数据位置, (2, n)

    for i in range(to_pred.shape[1]):

        r, c = to_pred[:, i]                        # r为用户索引，c为电影索引

        id = np.argsort(similarity_user[r])[::-1]   # 将用户以相似度从大到小排序
        id = id[1: n_keep + 1]                      # 获取相似度最大的几个用户，除自身
        rates = train_matrix[id, c]                 # 获取这几个用户对该电影的评分
        rates = rates[rates!=0]                     # 已评价的数据

        rate = np.mean(rates) if rates.shape[0] != 0 else 0
        pred_matrix[r, c] = rate

    return pred_matrix
    

def according_to_item(train_matrix, test_matrix, cols=80, n_keep=100):
  
    # 将矩阵SVD分解
    u, _, _ = np.linalg.svd(train_matrix)

    # 压缩原矩阵，A' = U[:. :k]^T A
    train_compressed_row = u.T[: cols].dot(train_matrix)    # cols    x N_ITEMS

    # 计算相似度矩阵
    similarity_item = get_cosine_similarity_matrix(train_compressed_row.T)

    # 预测
    pred_matrix = np.zeros_like(test_matrix)        # 保存预测结果
    to_pred = np.array(np.where(test_matrix != 0))  # 需要预测的数据位置, (2, n)

    for i in range(to_pred.shape[1]):

        r, c = to_pred[:, i]                        # r为用户索引，c为电影索引

        id = np.argsort(similarity_item[c])[::-1]   # 将电影以相似度从大到小排序
        id = id[1: n_keep + 1]                      # 获取相似度最大的几部电影，除自身
        rates = train_matrix[r, id]                 # 获取几部电影评分
        rates = rates[rates!=0]                     # 已评价的数据

        rate = np.mean(rates) if rates.shape[0] != 0 else 0
        pred_matrix[r, c] = rate

    return pred_matrix

if __name__ == "__main__":

    N_USERS, N_ITEMS = 943, 1682
    shape = (N_USERS, N_ITEMS)
    
    ## 载入数据
    train_raw = load('../origin/ml-100k/u1.base')
    train_matrix = get_matrix(train_raw, shape)
    test_raw  = load('../origin/ml-100k/u1.test')
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

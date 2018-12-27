%% 数据读取和参数设置
clear all; clc;
data = load("./数据/第4题数据/4a.mat"); %题目数据， data
data = data.data

N=8318; %数据点个数
DIS=10; %搜索点个数
points=data';
%% 创建 Kd-tree
kdtreeobj=KDTreeSearcher(points,'distance','euclidean');
%% PCA 方法计算三维点切平面的法向量
for j=1:1:8318
    xj=points(j,:);
    [idx,D]=knnsearch(kdtreeobj,xj,'K',DIS);%这里用的 knn 搜索，即 k 近邻点搜索
    X=[];
    %提取搜索到的点
    for i=1:1:DIS
        X(i,:)=points(idx(i),:);
    end
    %pca 方法得到 COEFF 三个特征向量以及 LATENT 特征值
    [COEFF, SCORE, LATENT] = pca(X);
    P=COEFF;
    %特征值越小，投影点之间的方差越小
    X1(:,j)=P(:,3);
end
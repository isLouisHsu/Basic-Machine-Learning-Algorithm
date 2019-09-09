%% ���ݶ�ȡ�Ͳ�������
clear all; clc;
data = load("./����/��4������/4a.mat"); %��Ŀ���ݣ� data
data = data.data

N=8318; %���ݵ����
DIS=10; %���������
points=data';
%% ���� Kd-tree
kdtreeobj=KDTreeSearcher(points,'distance','euclidean');
%% PCA ����������ά����ƽ��ķ�����
for j=1:1:8318
    xj=points(j,:);
    [idx,D]=knnsearch(kdtreeobj,xj,'K',DIS);%�����õ� knn �������� k ���ڵ�����
    X=[];
    %��ȡ�������ĵ�
    for i=1:1:DIS
        X(i,:)=points(idx(i),:);
    end
    %pca �����õ� COEFF �������������Լ� LATENT ����ֵ
    [COEFF, SCORE, LATENT] = pca(X);
    P=COEFF;
    %����ֵԽС��ͶӰ��֮��ķ���ԽС
    X1(:,j)=P(:,3);
end
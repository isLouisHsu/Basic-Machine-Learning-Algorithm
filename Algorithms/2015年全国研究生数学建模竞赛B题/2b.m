clear all;
clc;
%% 2015数模竞赛 第二题（b）
data = load("./数据/第2题数据/2b.mat")
data = data.data
[coeff,score,latent] = pca(data');   % 对数据进行pca分解
% COFFEE即是排序后特征向量矩阵(eigenvectors1) 
% LATENT即是排序后对应的特征值(eigenvalues) 
% SCORE即是未按特征值重要性级别排除删除特征值而计算出的主元

for i =1 : 300
    Y = data(:,i);
    % B = lasso(coeff(:,1:2),Y','Lambda',0.01);  % 选取前两个主分量作为主平面，运用lasso进行重建
    B = pinv(coeff(:,1:2)) * Y
    temp(i) = norm(Y-coeff(:,1:2)*B);
end

% 门限设为数据重建误差的均值，小于均值的视为不在平面以内
thre = mean(temp); 
[~,ind] = find(temp>thre);
[~,ind2] = find(temp<=thre);
newdata = data(:,ind);      % 在平面外的点
newdata2 = data(:,ind2);    % 在平面内的点
plot3(newdata(1,:),newdata(2,:),newdata(3,:),'*')
%plot3(newdata2(1,:),newdata2(2,:),newdata2(3,:),'r<')
hold on

% 继续做pca，找到主分量，并投影到二维，在二维平面上对两条直线进行聚类
[coe,~] = pca(newdata');
projdata = coe(:,1:2)'*newdata;
[~,ind3] = find(projdata(1,:)<0.15 & projdata(1,:)>-0.15);
 newdata3 = newdata(:,ind3);
plot3(newdata3(1,:),newdata3(2,:),newdata3(3,:),'b^')

[~,ind4] = find(projdata(2,:)<0.15 & projdata(2,:)>-0.15);
 newdata4= newdata(:,ind4);
plot3(newdata4(1,:),newdata4(2,:),newdata4(3,:),'k<')
grid on
clear all;
clc;
%% 2015��ģ���� �ڶ��⣨b��
data = load("./����/��2������/2b.mat")
data = data.data
[coeff,score,latent] = pca(data');   % �����ݽ���pca�ֽ�
% COFFEE���������������������(eigenvectors1) 
% LATENT����������Ӧ������ֵ(eigenvalues) 
% SCORE����δ������ֵ��Ҫ�Լ����ų�ɾ������ֵ�����������Ԫ

for i =1 : 300
    Y = data(:,i);
    % B = lasso(coeff(:,1:2),Y','Lambda',0.01);  % ѡȡǰ������������Ϊ��ƽ�棬����lasso�����ؽ�
    B = pinv(coeff(:,1:2)) * Y
    temp(i) = norm(Y-coeff(:,1:2)*B);
end

% ������Ϊ�����ؽ����ľ�ֵ��С�ھ�ֵ����Ϊ����ƽ������
thre = mean(temp); 
[~,ind] = find(temp>thre);
[~,ind2] = find(temp<=thre);
newdata = data(:,ind);      % ��ƽ����ĵ�
newdata2 = data(:,ind2);    % ��ƽ���ڵĵ�
plot3(newdata(1,:),newdata(2,:),newdata(3,:),'*')
%plot3(newdata2(1,:),newdata2(2,:),newdata2(3,:),'r<')
hold on

% ������pca���ҵ�����������ͶӰ����ά���ڶ�άƽ���϶�����ֱ�߽��о���
[coe,~] = pca(newdata');
projdata = coe(:,1:2)'*newdata;
[~,ind3] = find(projdata(1,:)<0.15 & projdata(1,:)>-0.15);
 newdata3 = newdata(:,ind3);
plot3(newdata3(1,:),newdata3(2,:),newdata3(3,:),'b^')

[~,ind4] = find(projdata(2,:)<0.15 & projdata(2,:)>-0.15);
 newdata4= newdata(:,ind4);
plot3(newdata4(1,:),newdata4(2,:),newdata4(3,:),'k<')
grid on
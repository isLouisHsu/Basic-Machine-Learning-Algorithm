%% 初始化
clear all; clc;
data = load("./数据/第4题数据/4a.mat"); %题目数据， data
data = data.data
load Xs10.mat; %pca 计算所有点的结果， X1
N=8318; %数据点个数
points=data';
Xs=X1';
%这是一个表示数据剩余要处理点的矩阵
%第四列代表映射的类别，第五列代表与源数据的映射位置
Drest=[points zeros(N,1) (1:N)'];
nowMax=1; %当前类别
%% Start
while size(Drest(Drest(:,4)==0))>0 %判断剩余矩阵的个数
    nowID=1;
    x0=Drest(nowID,1:3);
    v0=Xs(Drest(nowID,5),:); %直接从 pca 结果中提取法向量
    while 1
        %Kd-tree 的建立，使用欧式距离
        kdtreeobj=KDTreeSearcher(Drest(:,1:3),'distance','euclidean');
        %0.2 范围内搜索
        [idx,D]=rangesearch(kdtreeobj,x0,0.2);
        id=0;
        %遍历搜索到的点，并比较其法向量的夹角
        for i = 1:1:size(idx{1},2)
            id=idx{1}(i);
            v1=Xs(Drest(id,5),:);
            %为了调试方便采用角度制，自己写了一个求夹角的函数
            args = get_arg(v1,v0);
            if(args<5||args>175)
                points(Drest(id,5),4)=nowMax;
                Drest(id,4)=nowMax;
            end
        end
        %移除当前点
        Drest(nowID,:) = [];
        %搜索同类别下的第一个点
        if size(Drest(Drest(:,4)==nowMax))>0
            for i = 1:1:size(Drest,1)
                if(Drest(i,4)==nowMax)
                    nowID=i;
                    x0=Drest(i,1:3);
                    v0=Xs(Drest(i,5),:);
                    break;
                end
            end
        else
            %跳出 while 1，并将剩下的归为下一类， out 为统计结果
            out(nowMax,:)=[size(points(points(:,4)==nowMax),1) nowMax];
            nowMax=nowMax+1;
            break;
        end
    end
end
%% 绘制图像
figure(1);
plot3(points(:,1),points(:,2),points(:,3),'b.');
figure(2);
plot3(points(:,1),points(:,2),points(:,3),'y.');
%将得到的统计结果按最多到最少排列
hold on;
out=sortrows(out,-1);
for i=1:1:N
    if(points(i,4)==out(1,2))
        plot3(points(i,1),points(i,2),points(i,3),'r*');
    elseif(points(i,4)==out(2,2))
        plot3(points(i,1),points(i,2),points(i,3),'g*');
    elseif(points(i,4)==out(3,2))
        plot3(points(i,1),points(i,2),points(i,3),'b*');
    end
end
%将个数少的类别的点按就近原则归类
Dtemp=points;
Dgood=[];
for i = size(Dtemp,1):-1:1
    if( Dtemp(i,4)==out(1,2)||...
        Dtemp(i,4)==out(2,2)||...
        Dtemp(i,4)==out(3,2))
        Dgood=[Dgood;Dtemp(i,:)];
        Dtemp(i,:)=[];
    end
end
kdtreeobj=KDTreeSearcher(Dgood(:,1:3),'distance','euclidean');
for i = 1:1:size(Dtemp,1)
    [idx,D]=knnsearch(kdtreeobj,Dtemp(i,1:3),'K',1);
    Dtemp(i,4)=Dgood(idx(1),4);
end
Dgood=[Dgood;Dtemp];
figure(3);
plot3(points(:,1),points(:,2),points(:,3),'y.');
hold on;
for i = 1:1:N
    if(Dgood(i,4)==out(1,2))
        plot3(Dgood(i,1),Dgood(i,2),Dgood(i,3),'r*');
    elseif(Dgood(i,4)==out(2,2))
        plot3(Dgood(i,1),Dgood(i,2),Dgood(i,3),'g*');
    elseif(Dgood(i,4)==out(3,2))
        plot3(Dgood(i,1),Dgood(i,2),Dgood(i,3),'b*');
    end
end
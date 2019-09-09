%% ��ʼ��
clear all; clc;
data = load("./����/��4������/4a.mat"); %��Ŀ���ݣ� data
data = data.data
load Xs10.mat; %pca �������е�Ľ���� X1
N=8318; %���ݵ����
points=data';
Xs=X1';
%����һ����ʾ����ʣ��Ҫ�����ľ���
%�����д���ӳ�����𣬵����д�����Դ���ݵ�ӳ��λ��
Drest=[points zeros(N,1) (1:N)'];
nowMax=1; %��ǰ���
%% Start
while size(Drest(Drest(:,4)==0))>0 %�ж�ʣ�����ĸ���
    nowID=1;
    x0=Drest(nowID,1:3);
    v0=Xs(Drest(nowID,5),:); %ֱ�Ӵ� pca �������ȡ������
    while 1
        %Kd-tree �Ľ�����ʹ��ŷʽ����
        kdtreeobj=KDTreeSearcher(Drest(:,1:3),'distance','euclidean');
        %0.2 ��Χ������
        [idx,D]=rangesearch(kdtreeobj,x0,0.2);
        id=0;
        %�����������ĵ㣬���Ƚ��䷨�����ļн�
        for i = 1:1:size(idx{1},2)
            id=idx{1}(i);
            v1=Xs(Drest(id,5),:);
            %Ϊ�˵��Է�����ýǶ��ƣ��Լ�д��һ����нǵĺ���
            args = get_arg(v1,v0);
            if(args<5||args>175)
                points(Drest(id,5),4)=nowMax;
                Drest(id,4)=nowMax;
            end
        end
        %�Ƴ���ǰ��
        Drest(nowID,:) = [];
        %����ͬ����µĵ�һ����
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
            %���� while 1������ʣ�µĹ�Ϊ��һ�࣬ out Ϊͳ�ƽ��
            out(nowMax,:)=[size(points(points(:,4)==nowMax),1) nowMax];
            nowMax=nowMax+1;
            break;
        end
    end
end
%% ����ͼ��
figure(1);
plot3(points(:,1),points(:,2),points(:,3),'b.');
figure(2);
plot3(points(:,1),points(:,2),points(:,3),'y.');
%���õ���ͳ�ƽ������ൽ��������
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
%�������ٵ����ĵ㰴�ͽ�ԭ�����
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
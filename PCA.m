clear
%导入人脸拉为向量
dataname="AR";
if dataname=="AR"
    reshaped_faces=[];
    people_num=40;
    per_face_num=10;
    face_height=50;
    face_wide=40;
    for i=1:40
        for j=1:10
            if i<10
                a=imread(strcat('C:\Users\admin\Desktop\face\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif'));
            else
                a=imread(strcat('C:\Users\admin\Desktop\face\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif'));
            end
            b=reshape(a,[2000 1]);%拉成列向量
            b=double(b);%转换精度
            reshaped_faces=[reshaped_faces,b];%存储转换为列向量的人脸
        end
    end
elseif dataname=="face"
    reshaped_faces=[];
    people_num=15;
    per_face_num=11;
    face_height=100;
    face_wide=80;
    for i=1:15
        for j=1:11
            if i<10
                a=imread(strcat('C:\Users\admin\Desktop\face\face10080\subject0',num2str(i),'_',num2str(j),'.bmp'));
            else
                a=imread(strcat('C:\Users\admin\Desktop\face\face10080\subject',num2str(i),'_',num2str(j),'.bmp'));
            end
            b=reshape(a,[8000 1]);%拉成列向量
            b=double(b);%转换精度
            reshaped_faces=[reshaped_faces,b];%存储转换为列向量的人脸
        end
    end
else
    reshaped_faces=[];
    people_num=40;
    per_face_num=10;
    face_height=56;
    face_wide=46;
    for i=1:40
        for j=1:10
            a=imread(strcat('C:\Users\admin\Desktop\face\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));
            b=reshape(a,[2576 1]);%拉成列向量
            b=double(b);%转换精度
            reshaped_faces=[reshaped_faces,b];%存储转换为列向量的人脸
        end
    end
end
%取测试数据与训练数据
test_data=[];
train_data=[];
test_data_index=[];
train_data_index=[];
train_num=3;
test_num=per_face_num-train_num;
for i=1:people_num
    for j=1:per_face_num
        if  j>train_num
            test_data_index=[test_data_index,(i-1)*10+j];%取测试数据
        else
            train_data_index=[train_data_index,(i-1)*10+j];%取训练数据
        end
    end
end
test_data=reshaped_faces(:,test_data_index);
train_data=reshaped_faces(:,train_data_index);
%人脸零均值化
mean_face=mean(train_data,2,"omitmissing");
centered_faces=train_data-mean_face;
%求散度矩阵
cov_matrix=centered_faces*centered_faces';
[eigens,diag_matrix]=eig(cov_matrix);%分解散度矩阵得到特征向量矩阵和特征值的对角矩阵
eigens_value=diag(diag_matrix);%从对角矩阵获取特征值
[~,value_index]=sort(eigens_value,"descend");%根据特征值从大到小进行排序
sorted_eigens=eigens(:,value_index);
%展示投影
loc=1;
for i=1:10
    subplot(2,5,loc);
    loc=loc+1;
    fig=show_face(sorted_eigens(:,i),face_height,face_wide);
    if i==10
        waitfor(fig);
    end
end
%人脸重构
first_face=centered_faces(:,1);%取出第一个人的第一张人脸进行重构
loc=1;
for k=[10 20 50]
    eigen_faces=sorted_eigens(:,1:k);%取前k大特征向量进行重构
    rebuild_face=eigen_faces*(eigen_faces'*first_face)+mean_face;
    subplot(1,3,loc);
    loc=loc+1;
    fig=show_face(rebuild_face,face_height,face_wide);
    title(sprintf("dimension=%d",k));
    if k==50
        waitfor(fig);
    end
end
%人脸识别
%训练knn模型
index_train_faces=[];
index=1;
for i=1:people_num*train_num
    index_train_faces=[index_train_faces,index];%创建训练集人脸编号索引
    if mod(i,train_num)==0
        index=index+1;
    end
end
index_test_faces=[];
index=1;
for i=1:people_num*test_num
    index_test_faces=[index_test_faces,index];%创建测试集人脸编号索引
    if mod(i,test_num)==0
        index=index+1;
    end
end
max_accuracy=0;
max_i=0;
for k=1:10
    for i=10:10:160
        eigen_faces=sorted_eigens(:,1:i);%取前i个特征向量以降到i维
        projected_train_data=eigen_faces'*(train_data-mean_face);%将训练集降到i维
        projected_test_data=eigen_faces'*(test_data-mean_face);%将测试集降到i维
        % projected_train_data=train_data;%对照组，不对训练集采取降维
        % projected_test_data=test_data;%对照组，不对测试集采取降维
        distance=[];
        identified_test_data=[];
        for j=1:people_num*test_num
            distance=sqrt(sum((projected_test_data(:,j)-projected_train_data).^2));%计算测试样本与每个训练样本之间的距离
            index=[];
            [~,index]=sort(distance,"ascend");%将距离升序排序好，并记录下各距离对应的训练样本的编号索引
            sorted_distance_index=index_train_faces(index);%获取排完序后训练样本对应的编号
            identified_test_data(j)=mode(sorted_distance_index(1:k));%获取距离最小的前k个训练样本的编号中出现次数最多的，将该测试样本对应到该类
        end
        accuracy=sum(identified_test_data==index_test_faces)/(test_num*people_num);%计算识别率
        if max_accuracy<accuracy
            max_accuracy=accuracy;%更新最大识别率
            max_i=i;%更新最大识别率对应的维数
        end
    end
end
%人脸图像降维与可视化
for i=2:3
    eigen_faces=sorted_eigens(:,1:i);
    show_test_data=eigen_faces'*(test_data-mean_face);
    color=[];
    for j=1:test_num*people_num
        color = [color floor((j-1)/4)*5];
    end
    if i==2
        waitfor(scatter(show_test_data(1, :), show_test_data(2, :), [], color));
    else
        waitfor(scatter3(show_test_data(1, :), show_test_data(2, :),show_test_data(3, :), [], color));
    end
end
%自定义函数显示人脸
function fig=show_face(face,face_height,face_wide)
    fig=imshow(mat2gray(reshape(face,[face_height,face_wide])));
end
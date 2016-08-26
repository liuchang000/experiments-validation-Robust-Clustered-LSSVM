clear
clc

load UCI.mat
N=5400;
x=data(1:N,1:end-1);
y=data(1:N,end);
X=[x,y];

[idx,ctrs,s,D]=kmeans(X,3,'dist','sqeuclidean');

x1=[x(idx==1,1),x(idx==1,2),x(idx==1,3),x(idx==1,4)];
x2=[x(idx==2,1),x(idx==2,2),x(idx==2,3),x(idx==2,4)];
x3=[x(idx==3,1),x(idx==3,2),x(idx==3,3),x(idx==3,4)];
y1=y(idx==1);
y2=y(idx==2);
y3=y(idx==3);

Tn1=round(1/3*length(x1));
Tn2=round(1/3*length(x2));
Tn3=round(1/3*length(x3));
x_1=x1(1:end-Tn1,:);
x_2=x2(1:end-Tn2,:);
x_3=x3(1:end-Tn3,:);
train_x=[x_1;x_2;x_3];
y_1=y1(1:end-Tn1);
y_2=y2(1:end-Tn2);
y_3=y3(1:end-Tn3);
train_y=[y_1;y_2;y_3];
Rtest_y1=y1(length(y_1)+1:end);
Rtest_y2=y2(length(y_2)+1:end);
Rtest_y3=y3(length(y_3)+1:end);

gam=1;
sig2=100;

test_x1=x1(length(x_1)+1:end,:);
test_x2=x2(length(x_2)+1:end,:);
test_x3=x3(length(x_3)+1:end,:);
test_x=[test_x1;test_x2;test_x3];

[alpha,b]=trainlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'});

test_y_1=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x1);
test_y_2=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x2);
test_y_3=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x3);

error2=abs([test_y_1;test_y_2;test_y_3]-[Rtest_y1;Rtest_y2;Rtest_y3]);
test_RMSE2=sqrt(sum(error2.^2/length(error2)))
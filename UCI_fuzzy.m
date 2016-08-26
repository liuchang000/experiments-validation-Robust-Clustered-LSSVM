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

ave_x=sum(train_x)/length(train_x);
dis=0;
for i=1:length(train_x)
    dis(i)=norm(train_x(i,:)-ave_x)/10;
end
Dis=(1-exp(-dis))./(1+exp(-dis));

Y=[zeros(3,1);y_1;y_2;y_3];
test_x1=x1(length(x_1)+1:end,:);
test_x2=x2(length(x_2)+1:end,:);
test_x3=x3(length(x_3)+1:end,:);
test_x=[test_x1;test_x2;test_x3];
gamma=30;
sigma2=100;
QQ=0;

for i=1:length(train_x)
    for j=1:length(train_x)
        QQ(i,j)=rbf(train_x(i,:),train_x(j,:),sigma2);
    end
end

q=diag(Dis/gamma);
Q=QQ+q;
B=ones(1,length(train_x));
T=[0 B;B' Q];
G=inv(T'*T)*T'*[0;y_1;y_2;y_3];

P=0;
for i=1:length(test_x)
    for j=1:length(train_x)
        P(i,j)=rbf(test_x(i,:),train_x(j,:),sigma2);
    end
end

test_y=[ones(length(test_x),1),P]*G;

error1=abs(test_y-[Rtest_y1;Rtest_y2;Rtest_y3]);
test_RMSE1=sqrt(sum(error1.^2/length(error1)))
clear
clc

load PRESS.mat

a=randperm(length(yaji));
yaji=yaji(a,:);

x=yaji(:,[2:3,6]);
y=yaji(:,1);
X=[x,y];

[idx,ctrs,s,D]=kmeans(X,3,'dist','sqeuclidean');

x1=[x(idx==1,1),x(idx==1,2),x(idx==1,3)];
x2=[x(idx==2,1),x(idx==2,2),x(idx==2,3)];
x3=[x(idx==3,1),x(idx==3,2),x(idx==3,3)];
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

dd=min(D,[],2)*1000;
d=(1-exp(-dd))./(1+exp(-dd));
d1=d(idx==1);
d2=d(idx==2);
d3=d(idx==3);
d_1=d1(1:end-Tn1,:);
d_2=d2(1:end-Tn2,:);
d_3=d3(1:end-Tn3,:);

Y=[zeros(3,1);y_1;y_2;y_3];
gamma=20;
sigma2=1;

K11=0;K22=0;K33=0;K12=0;K13=0;K23=0;
for i=1:length(x_1)
    for j=1:length(x_1)
        K11(i,j)=rbf(x_1(i,:),x_1(j,:),sigma2);
    end
end
for i=1:length(x_2)
    for j=1:length(x_2)
        K22(i,j)=rbf(x_2(i,:),x_2(j,:),sigma2);
    end
end
for i=1:length(x_3)
    for j=1:length(x_3)
        K33(i,j)=rbf(x_3(i,:),x_3(j,:),sigma2);
    end
end
for i=1:length(x_1)
    for j=1:length(x_2)
        K12(i,j)=rbf(x_1(i,:),x_2(j,:),sigma2);
    end
end
for i=1:length(x_1)
    for j=1:length(x_3)
        K13(i,j)=rbf(x_1(i,:),x_3(j,:),sigma2);
    end
end
for i=1:length(x_2)
    for j=1:length(x_3)
        K23(i,j)=rbf(x_2(i,:),x_3(j,:),sigma2);
    end
end
KK=[K11 K12 K13;K12' K22 K23;K13' K23' K33];
k=diag([d_1;d_2;d_3]/gamma);
K=KK+k;
A=zeros(3,length(x_1)+length(x_2)+length(x_3));
A(1,1:length(x_1))=1;
A(2,1+length(x_1):length(x_1)+length(x_2))=1;
A(3,1+length(x_1)+length(x_2):end)=1;
H=[zeros(3,3) A;A' K];
W=inv(H'*H)*H'*Y;

Ktest_x1=0;Ktest_x2=0;Ktest_x3=0;
test_x1=x1(length(x_1)+1:end,:);
for i=1:length(test_x1)
    for j=1:length(train_x)
        Ktest_x1(i,j)=rbf(test_x1(i,:),train_x(j,:),sigma2);
    end
end
test_x2=x2(length(x_2)+1:end,:);
for i=1:length(test_x2)
    for j=1:length(train_x)
        Ktest_x2(i,j)=rbf(test_x2(i,:),train_x(j,:),sigma2);
    end
end
test_x3=x3(length(x_3)+1:end,:);
for i=1:length(test_x3)
    for j=1:length(train_x)
        Ktest_x3(i,j)=rbf(test_x3(i,:),train_x(j,:),sigma2);
    end
end

HH=[A' KK];
traindata_predict=HH*W;

test_A1=zeros(length(test_x1),3);
test_A1(:,1)=1;
hh1=[test_A1,Ktest_x1];
test_y1=hh1*W;

test_A2=zeros(length(test_x2),3);
test_A2(:,2)=1;
hh2=[test_A2,Ktest_x2];
test_y2=hh2*W;

test_A3=zeros(length(test_x3),3);
test_A3(:,3)=1;
hh3=[test_A3,Ktest_x3];
test_y3=hh3*W;

error=abs([test_y1;test_y2;test_y3]-[Rtest_y1;Rtest_y2;Rtest_y3]);
test_RMSE=sqrt(sum(error.^2/length(error)))

figure(1)
plot(train_y,'r')
hold on
plot(traindata_predict,'g')
legend('real output','model output','location','NW')

figure(2)
plot([Rtest_y1;Rtest_y2;Rtest_y3],'r')
hold on
plot([test_y1;test_y2;test_y3],'g')
legend('real output','model output','location','NW')

real=[Rtest_y1;Rtest_y2;Rtest_y3];
model=[test_y1;test_y2;test_y3];

n=randperm(length(real));
figure(3)
plot(real(n(1:100)),'r','linewidth',2.5)
hold on
plot(model(n(1:100)),'g--','linewidth',2.2)
legend('real output','model output','location','NW')
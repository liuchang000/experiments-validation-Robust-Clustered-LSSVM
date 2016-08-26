clear
clc

load TEMPERATURE.mat

data=UT;
x=[data(2:end,1:end-1),data(1:end-1,end)];
y=data(2:end,end);
X=[x,y];
[idx,ctrs,s,D]=kmeans(X,3,'dist','sqeuclidean');

x1=[x(idx==1,1),x(idx==1,2),x(idx==1,3),x(idx==1,4),x(idx==1,5)];
x2=[x(idx==2,1),x(idx==2,2),x(idx==2,3),x(idx==2,4),x(idx==2,5)];
x3=[x(idx==3,1),x(idx==3,2),x(idx==3,3),x(idx==3,4),x(idx==3,5)];
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

dd=min(D,[],2)/6;
d=(1-exp(-dd))./(1+exp(-dd));
d1=d(idx==1);
d2=d(idx==2);
d3=d(idx==3);
d_1=d1(1:end-Tn1,:);
d_2=d2(1:end-Tn2,:);
d_3=d3(1:end-Tn3,:);

ave_x=sum(train_x)/length(train_x);
dis=0;
for i=1:length(train_x)
    dis(i)=norm(train_x(i,:)-ave_x)/10;
end
Dis=(1-exp(-dis))./(1+exp(-dis));

Y=[zeros(3,1);y_1;y_2;y_3];
gamma=20;
sigma2=3;

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

%%  fuzzy
QQ=0;
gamm=50;
sigm2=2;
for i=1:length(train_x)
    for j=1:length(train_x)
        QQ(i,j)=rbf(train_x(i,:),train_x(j,:),sigm2);
    end
end
q=diag(Dis/gamm);
Q=QQ+q;
B=ones(1,length(train_x));
T=[0 B;B' Q];
G=inv(T'*T)*T'*[0;y_1;y_2;y_3];

test_x=[test_x1;test_x2;test_x3];
P=0;
for i=1:length(test_x)
    for j=1:length(train_x)
        P(i,j)=rbf(test_x(i,:),train_x(j,:),sigm2);
    end
end
test_y=[ones(length(test_x),1),P]*G;

%%  proposed
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

%%  clustered
gam_1=60;
sig2_1=1;
[alpha1,b1]=trainlssvm({x_1,y_1,'f',gam_1,sig2_1,'RBF_kernel'});
[alpha2,b2]=trainlssvm({x_2,y_2,'f',gam_1,sig2_1,'RBF_kernel'});
[alpha3,b3]=trainlssvm({x_3,y_3,'f',gam_1,sig2_1,'RBF_kernel'});

test_y11=simlssvm({x_1,y_1,'f',gam_1,sig2_1,'RBF_kernel'},{alpha1,b1},test_x1);
test_y22=simlssvm({x_2,y_2,'f',gam_1,sig2_1,'RBF_kernel'},{alpha2,b2},test_x2);
test_y33=simlssvm({x_3,y_3,'f',gam_1,sig2_1,'RBF_kernel'},{alpha3,b3},test_x3);

%%  traditional
gam=1000;
sig2=0.5;
[alpha,b]=trainlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'});

test_y_1=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x1);
test_y_2=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x2);
test_y_3=simlssvm({train_x,train_y,'f',gam,sig2,'RBF_kernel'},{alpha,b},test_x3);

error=abs([test_y1;test_y2;test_y3]-[Rtest_y1;Rtest_y2;Rtest_y3]);
error1=abs(test_y-[Rtest_y1;Rtest_y2;Rtest_y3]);
error2=abs([test_y_1;test_y_2;test_y_3]-[Rtest_y1;Rtest_y2;Rtest_y3]);
error3=abs([test_y11;test_y22;test_y33]-[Rtest_y1;Rtest_y2;Rtest_y3]);

test_RMSE=sqrt(sum(error.^2/length(test_y)));
test_RMSE1=sqrt(sum(error1.^2/length(test_y)));
test_RMSE2=sqrt(sum(error2.^2/length(test_y)));
test_RMSE3=sqrt(sum(error3.^2/length(test_y)));

%% 
RMSE=[test_RMSE;test_RMSE1;test_RMSE3;test_RMSE2]
% Proposed method;Fuzzy LS-SVM;Clustered LS-SVM;Traditional LS-SVM

real=[Rtest_y1;Rtest_y2;Rtest_y3];
model=[test_y1;test_y2;test_y3];

n=randperm(length(real));
figure(1)
plot(real(n(1:100)),'r')
hold on
plot(model(n(1:100)),'g--')
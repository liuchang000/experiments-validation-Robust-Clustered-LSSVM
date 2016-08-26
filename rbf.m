function Y=rbf(x,y,sig2)
Y=exp(-(x-y)*(x-y)'/sig2);
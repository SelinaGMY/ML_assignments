function [y,cov] = GP(x,k)
    cov = covkernel(x,x,k);
    l = size(x,2);
    y = mvnrnd(zeros(1,l),cov);
end
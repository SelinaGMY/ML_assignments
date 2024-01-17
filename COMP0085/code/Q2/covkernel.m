function cov = covkernel(x1,x2,k)
    l1 = size(x1,2);
    l2 = size(x2,2);
    cov = combvec(x1,x2);
    cov = arrayfun(@(i)k(cov(1,i),cov(2,i)),1:(l1*l2));
    cov = reshape(cov,l1,l2);
end
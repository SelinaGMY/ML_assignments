function[A,C,Q,R,Loglik] = LGSSM_EM(X,A,C,Q,R,Q_init,y_init,iter)

T = size(X,2);
Loglik = ones(iter,1);

for i = 1:iter+1
    % E-step
    [Y,V,Vj,L] = ssm_kalman(X,y_init,Q_init,A,Q,C,R,'smooth');
    Loglik(i) = sum(L);

    % M-step
    cellsum = @(C)(sum(cat(3,C{:}),3));
    
    YYt = Y*Y' + cellsum(V);
    YYt1 = YYt - Y(:,end)*Y(:,end)' - cellsum(V(end));
    YYt2 = YYt - Y(:,1)*Y(:,1)' - cellsum(V(1));
    Yt2Yt1 = Y(:,2:end)*Y(:,1:end-1)' + cellsum(Vj);

    C = X*Y' / YYt;
    A = Yt2Yt1 / YYt1;
    R = 1/T * (X*X' - X*Y'*C');
    Q = 1/(T-1) * (YYt2 - Yt2Yt1*A');
    y_init = Y(:,1) ;
    Q_init = V{1};

    
end

end
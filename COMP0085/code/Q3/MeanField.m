function [lambda,F] = MeanField(X,mu,sigma,pie,lambda0,maxsteps)
    [~,K] = size(lambda0);
    [N,D] = size(X);
    lambda = lambda0;
    epsilon = 10^(-8);
    F = -Inf;
    
    % avoid numerical issue
    pie(pie==1) = 1 - eps;
    pie(pie==0) = eps;
    
    for i=1:maxsteps
        % update ES, ESS
        for j=1:K
            notj = [1:(j-1),(j+1):K];
            ES = lambda;
            loglik = log(pie(j)/(1-pie(j))) - 1/(2*sigma^2)*(mu(:,j)'*mu(:,j)-2*X*mu(:,j) ...
                +2*ES(:,notj)*mu(:,notj)'*mu(:,j));
            infty = exp(loglik) == Inf;
            fty = exp(loglik) < Inf;
            lambda(fty,j) = 1 ./ (1+exp(-loglik(fty)));
            lambda(infty,j) = 1;
        end
        ES = lambda;
        ESS = lambda' * lambda;
        for i = 1: size (lambda ,2)
            ESS(i,i) = sum(lambda(:,i));
        end
        
        % avoid numerical issue
        lambda(lambda==0) = eps ;
        lambda(lambda==1) = 1 - eps ;

        % free energy
        Hs = -sum(lambda.*log(lambda) + (1-lambda).*log(1-lambda),2);
        currentF = -N*D/2*log(2*pi*sigma^2) - 1/(2*sigma^2)*(sum(sum(X.*X))-2*sum(sum((ES*mu').*X)) ...
            +sum(sum(ESS.*(mu'*mu)))) + sum(ES * log(pie./(1-pie))' + sum(log(1-pie))) + sum(Hs);

        if currentF - F < epsilon
            break
        end
        F = currentF;
    end
end






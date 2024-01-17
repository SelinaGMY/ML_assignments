function [lambda,Message,F] = LoopyBP(X,mu,sigma,pie,Message0,maxsteps)
    [N,D] = size(X);
    [~,K,~] = size(Message0);
    Message = zeros(size(Message0));
    F = -inf;
    epsilon = 10^(-8);
    
    % natural parameter of fi(si): bi
    f = zeros(N,K);
    for n=1:N
       f(n,:) = log(pie./(1-pie)) + X(n,:)*mu/(sigma^2) - diag(mu'*mu)'/(2*sigma^2);
    end

    for iter=1:maxsteps
        for n=1:N
            message_n = Message0(:,:,n);
            for i=1:K
                for j=(i+1):K
                    alpha = 0.5;
                    % update wji
                    Wij = - mu(:,i)'*mu(:,j)/(sigma^2);
                    nj = f(n,j) + sum(message_n(:,j)) - message_n(i,j);
                    wji = (exp(Wij+nj)+1)/(exp(nj)+1);
                    % message_n(j,i) = log(wji);
                    message_n(j,i) = alpha*message_n(j,i) + (1-alpha)*log(wji);

                    %update wij
                    Wji = - mu(:,j)'*mu(:,i)/(sigma^2);
                    ni = f(n,i) + sum(message_n(:,i)) - message_n(j,i);
                    wij = (exp(Wji+ni)+1)/(exp(ni)+1);
                    % message_n(i,j) = log(wij);
                    message_n(i,j) = alpha*message_n(i,j) + (1-alpha)*log(wij);
                end
            end
            Message(:,:,n) = message_n;
        end
        
        lambda = zeros(N,K);
        for n=1:N
            p =  f(n,:)+sum(Message(:,:,n));
            lambda(n,:) = 1./(1+exp(-p));
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

        diff = max(max(max(Message-Message0)));
        
        if diff < epsilon
            break
        end
        F = currentF;
        Message0 = Message;
    
    end
end









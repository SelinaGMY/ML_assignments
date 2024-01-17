function [mu,sigma,pie,lambda] = LearnBinFactors(X,K,iterations)
    [N,D] = size(X);
    F = -inf;
    epsilon = 10^(-8);
    maxsteps = 50;

    % initialisation
    lambda0 = rand(N,K);
    ES = lambda0;
    ESS = zeros(N,K,K);
    for n=1:N
        tmp = lambda0(n,:)'*lambda0(n,:);
        tmp(logical(eye(K)))=lambda0(n,:);
        ESS(n,:,:) = tmp;
    end
    [mu, sigma, pie] = MStep(X,ES,ESS);
    Message0 = rand(K,K,N);
    for n=1:N
        b = Message0(:,:,n);
        b = b-diag(diag(b));
        Message0(:,:,n) = b;
    end
    
    Ftotal = zeros(1,iterations);
    for j=1:iterations
        % E-step
        [lambda,Message,currentF] = LoopyBP(X,mu,sigma,pie,Message0,maxsteps);
        Ftotal(:,j) = currentF;

        % M-step
        ES = lambda;
        ESS = zeros(N,K,K);
        for n=1:N
            tmp = lambda(n,:)'*lambda(n,:);
            tmp(logical(eye(K)))=lambda(n,:);
            ESS(n,:,:) = tmp;
        end
        [mu, sigma, pie] = MStep(X,ES,ESS);

        if currentF - F < epsilon
            break
        end
        F = currentF;
        Message0 = Message;

    end
    figure
    plot(Ftotal);xlabel("iterations");ylabel("free energy")
end





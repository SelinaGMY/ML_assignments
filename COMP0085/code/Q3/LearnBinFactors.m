function [mu, sigma, pie, lambda] = LearnBinFactors(X,K,iterations)
    [N,~]=size(X);
    epsilon = 10^(-8);
    F = -inf;
    maxsteps = 50;

    % initialise
    lambda0 = rand(N,K);
    ES = lambda0;
    ESS = zeros(N,K,K);
    for n=1:N
        tmp = lambda0(n,:)'*lambda0(n,:);
        tmp(logical(eye(K)))=lambda0(n,:);
        ESS(n,:,:) = tmp;
    end
    [mu, sigma, pie] = MStep(X,ES,ESS);

    % EM
    Ftotal = zeros(1,iterations);
    for j=1:iterations
        % E-step
        [lambda,F_new] = MeanField(X,mu,sigma,pie,lambda0,maxsteps);
        Ftotal(j) = F_new;

        % M-step
        ES = lambda;
        ESS = zeros(N,K,K);
        for n=1:N
            tmp = lambda(n,:)'*lambda(n,:);
            tmp(logical(eye(K)))=lambda(n,:);
            ESS(n,:,:) = tmp;
        end
        [mu, sigma, pie] = MStep(X,ES,ESS);
        
        if F_new-F < epsilon
            break;
        end
        F = F_new;
        lambda0 = lambda;
    end
    
    figure;
    plot(Ftotal); xlabel("iterations");ylabel("free energy")
end

function F = FreeEnergy(ES,ESS,Hs,X,pie,mu,sigma)
    [N,D] = size(X);
    % avoid numerical issue
    pie(pie==1) = 1 - eps;
    pie(pie==0) = eps;
    
    % compute free energy
    F = -N*D/2*log(2*pi*sigma^2) - 1/(2*sigma^2)*(sum(sum(X.*X))-2*sum(sum((ES*mu').*X)) ...
            +sum(sum(ESS.*(mu'*mu)))) + sum(ES * log(pie./(1-pie))' + sum(log(1-pie))) + sum(Hs);

end
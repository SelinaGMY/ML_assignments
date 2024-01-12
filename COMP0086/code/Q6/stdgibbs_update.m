function [zi,theta,phi,Adk,Bkw,Mk] = stdgibbs_update(zi,theta,phi,Adk,Bkw,Mk,...
        I,D,K,W,di,wi,ci,citest,Id,Iw,Nd,alpha,beta);
% standard gibbs update

Adk = zeros(size(Adk)); 
Bkw = zeros(size(Bkw));

for ii = 1:I
    % conditional distribution
    pii = theta(di(ii),:)' .* phi(:,wi(ii)); 
    pii = pii / sum(pii);  
    
    % resample zi{ii}
    s = mnrnd(1,pii,ci(ii));
    [~,index] = max(s,[],2);
    zi{ii} = index';   
    
    % update Adk & Bkw
    for k=1:K
        Adk(di(ii),k)= Adk(di(ii),k) + sum(zi{ii}==k);
        Bkw(k,wi(ii))= Bkw(k,wi(ii)) + sum(zi{ii}==k);
    end
end

Nd = sum(Adk,2);
Mk = sum(Bkw,2);

% resample theta
for d=1:D
    theta(d,:) = dirichrnd(1,alpha + Adk(d,:)); 
end

% resample phi
for k=1:K
    phi(k,:) = dirichrnd(1,beta + Bkw(k,:));
end

end


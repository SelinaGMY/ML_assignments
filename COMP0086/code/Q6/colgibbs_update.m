function [zi,Adk,Bkw,Mk] = gibbs_update(zi,Adk,Bkw,Mk,...
        I,D,K,W,di,wi,ci,citest,Id,Iw,Nd,alpha,beta);
% collapsed gibbs update

Adk = zeros(size(Adk)); 
Bkw = zeros(size(Bkw));

for ii = 1:I
    pii = zeros(1,K); 
    for k=1:K
        % conditional prob
        pii(k) = (alpha+Adk(di(ii),k)-sum(zi{ii}==k)) * (beta+Bkw(k,wi(ii))-sum(zi{ii}==k))...
           /(K*alpha+Nd(di(ii)) - sum(zi{ii}==k))/(W*beta+Mk(k) - sum(zi{ii}==k));
        % update Adk & Bkw
        Adk(di(ii),k)= Adk(di(ii),k)+sum(zi{ii}==k);
        Bkw(k,wi(ii))= Bkw(k,wi(ii))+sum(zi{ii}==k);
    end
    pii = pii/sum(pii);
    
    % resample zi{ii}
    s = mnrnd(1,pii,ci(ii));
    [~,index]=max(s,[],2);
    zi{ii} = index';    
end

Nd = sum(Adk,2);
Mk = sum(Bkw,2);

end
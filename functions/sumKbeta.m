function Kaux=sumKbeta(K,beta)

if ~isstruct(K)
    ind=find(beta);
    Kaux=zeros(size(K(:,:,1)));
    N=length(ind);
    for j=1:N
        Kaux=Kaux+ beta(ind(j))*K(:,:,ind(j));
    end
else
    if size(beta,1)>1;
        beta=beta';
    end;
if isa(K.data,'single');
    Kaux=devectorize_single(K.data*beta');
else
    Kaux=devectorize(K.data*beta');
end;
end;
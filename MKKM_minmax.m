function [Hstar,Sigma,obj] = MKKM_minmax(KH,numclass,option)

numker = size(KH,3);
Sigma = ones(numker,1)/numker;

% KHP = zeros(num,num,numker);
% for p = 1:numker
%     KHP(:,:,p) = myLocalKernel(KH,tau,p);
% end
% KH = KHP;
% clear KHP
%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-2;
end
if ~isfield(option,'goldensearchmax')
    optiongoldensearchmax=1e-8;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end

%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
M = zeros(numker);
for p =1:numker
    for q = p:numker
        M(p,q) = trace(KH(:,:,p)*KH(:,:,q));
    end
end
M = M+M'-diag(diag(M));
M = (M+M')/2;

nloop = 1;
loop = 1;
goldensearch_deltmaxinit = option.goldensearch_deltmax;
%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Kmatrix = sumKbeta(KH,Sigma.^2);
[Hstar,obj1]= mykernelkmeans(Kmatrix,numclass);
obj(nloop) = obj1;
[grad] = MKKMGrad(KH,Hstar,Sigma);

Sigmaold  = Sigma;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop
    nloop = nloop+1;
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = MKKMupdate(KH,Sigmaold,grad,obj(nloop-1),numclass,option);
%     %-------------------------------
%     % Numerical cleaning
%     %-------------------------------
   Sigma(find(abs(Sigma<option.numericalprecision)))=0;
   Sigma = Sigma/sum(Sigma);

    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision &&...
            option.goldensearch_deltmax > optiongoldensearchmax
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end
    
    [grad] = MKKMGrad(KH,Hstar,Sigma);
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if  max(abs(Sigma-Sigmaold))<option.seuildiffsigma
        loop = 0;
        fprintf(1,'variation convergence criteria reached \n');
    end
    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------
    Sigmaold  = Sigma;
end
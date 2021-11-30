function [Hstar,Sigma,obj] = simpleMKKM(KH,numclass,option)

numker = size(KH,3);
Sigma = ones(numker,1)/numker;


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

nloop = 1;
loop = 1;
goldensearch_deltmaxinit = option.goldensearch_deltmax;

%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Kmatrix = sumKbeta(KH,Sigma.^2);
[Hstar,obj1]= mykernelkmeans(Kmatrix,numclass);
obj(nloop) = obj1;
% [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
[grad] = simpleMKKMGrad(KH,Hstar,Sigma);

Sigmaold  = Sigma;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop
    nloop = nloop+1;
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = simpleMKKMupdate(KH,Sigmaold,grad,obj(nloop-1),numclass,option);
    
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision &&...
            option.goldensearch_deltmax > optiongoldensearchmax
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end
    
    [grad] = simpleMKKMGrad(KH,Hstar,Sigma);
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
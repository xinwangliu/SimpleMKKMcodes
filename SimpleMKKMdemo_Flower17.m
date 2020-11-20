clear
clc
warning off;

path = pwd;
addpath(genpath(path));
dataName = 'flower17'; %%% flower17; flower102; CCV;
%% caltech101_nTrain5_48
%% proteinFold; UCI_DIGIT
%% SensITVehicle_1500sample_2view_3cluster
%% caltech101_mit_Kmatrix
%%% Handwritten_numerals;
% Caltech101-20; Caltech101-7
load([path,'/datasets/',dataName,'_Kmatrix'],'KH','Y');
Y(Y==-1)=2;
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KH = kcenter(KH);
KH = knorm(KH);
options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-16;   % numerical precision weights below this value
% are set to zero
%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=500;             % maximal number of iteration
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one

options.miniter=0;                 % minimal number of iterations
options.threshold = 1e-4;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qnorm = 2;

% %%%%%%%%---Average---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sigma1 = ones(numker,1)/numker;
avgKer  = mycombFun(KH,Sigma1);
[H_normalized1] = mykernelkmeans(avgKer, numclass);
[res_mean(:,1),res_std(:,1)] = myNMIACCV2(H_normalized1,Y,numclass);

%%%%---MKKM-----%%%
[H_normalized2,Sigma2,obj2] = mkkmeans_train(KH,numclass,qnorm);
[res_mean(:,2),res_std(:,2)] = myNMIACCV2(H_normalized2,Y,numclass);

%%--- The Proposed SimpleMKKM----
[H_normalized8,Sigma8,obj8] = MKKM_minmax(KH,numclass,options);
[res_mean(:,3),res_std(:,3)] = myNMIACCV2(H_normalized8,Y,numclass);
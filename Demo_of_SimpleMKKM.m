clear
clc
warning off;

path = './';
addpath(genpath(path));
dataName = '*';%'flower17','flower102','proteinFold', 'CCV','UCI_DIGIT','caltech101_nTrain30_48'
load(['*\',dataName,'_Kmatrix'],'KH','Y');

%% initialization
numclass = length(unique(Y));
Y(Y<1) = numclass;
numker = size(KH,3);
num = size(KH,1);
KH = kcenter(KH);
KH = knorm(KH);
options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-3; % initial precision of golden section search
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

%% ---- Multiple Kernel K-Means(MKKM)----- %%
tic;
[H_normalized0,Sigma0,obj0] = mkkmeans_train(KH,numclass);
[res_mean0,res_std0] = myNMIACCV2(H_normalized0,Y,numclass);
timecost0 = toc;

%% --- The Proposed SimpleMKKM(SMKKM)---- %%
tic;
[H_normalized,Sigma,obj] = simpleMKKM(KH,numclass,options);
[res_mean,res_std] = myNMIACCV2(H_normalized,Y,numclass);
timecost = toc;


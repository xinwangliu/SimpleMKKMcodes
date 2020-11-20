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
% 
% % %%%%%%%%%%%%---LMKKM (NIPS-2014)---%%%%%%%%%%%%%%%%%%%%%%%
% % [H_normalized3,obj3]= lmkkmeans_train(KH, numclass);
% % [res_mean(:,3),res_std(:,3)]= myNMIACCV2(H_normalized3,Y,numclass);
% 
% % %%%%%%%%%%%%%----ONKC (AAAI 2017)----------%%%%%%%%%%%%%%%%%%%%%%%
% rhoset4 = 2.^[0];
% lambdaset4 = 2.^[0];
% [H_normalized4,Sigma4,G4,obj4] = myoptimalNeighborhoodkernelclustering(KH,M,numclass,rhoset4,lambdaset4);
% [res_mean(:,4),res_std(:,4)]= myNMIACCV2(H_normalized4,Y,numclass);
% 
% %%%%%%%%%---MKKM-MiR(AAAI-16)----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lambdaset5 = 2.^[-1];
% [H_normalized5,Sigma5,obj5] = myregmultikernelclustering(KH,M,numclass,lambdaset5);
% [res_mean(:,5),res_std(:,5)] = myNMIACCV2(H_normalized5,Y,numclass);
% 
% % % %%%%%---LKAM (IJCAI-2016)--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % lambdaset6 = 2.^[0];
% % tauset6 = [0.5];
% % numSel = round(tauset6*num);
% % NS6 = genarateNeighborhood(avgKer,numSel);
% % %%--Calculate Neighborhood--%%%%%%
% % A6 = zeros(num);
% % for i =1:num
% %     A6(NS6(:,i),NS6(:,i)) = A6(NS6(:,i),NS6(:,i))+1;
% % end
% % HE6 = calHessian(KH,NS6,1);
% % [H_normalized6,Sigma6,obj6] = mylocalizedregmultikernelclustering(KH,numclass,qnorm,HE6,A6,lambdaset6);
% % [res_mean(:,6),res_std(:,6)]= myNMIACCV2(H_normalized6,Y,numclass);
% 
% %%%%%%%%%---(IJCAI-19)----%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lambda7 = 2.^[1]; 
% [H_normalized7,WP7,Sigma7,obj7] = myLateFusionMKC_lambda(KH,numclass,qnorm,lambda7);
% [res_mean(:,7),res_std(:,7)] = myNMIACCV2(H_normalized7,Y,numclass);

%%--- The Proposed SimpleMKKM----
[H_normalized8,Sigma8,obj8] = MKKM_minmax(KH,numclass,options);
[res_mean(:,8),res_std(:,8)] = myNMIACCV2(H_normalized8,Y,numclass);
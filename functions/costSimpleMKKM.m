function [cost,Hstar] = costSimpleMKKM(KH,StepSigma,DirSigma,Sigma,numclass)

global nbcall
nbcall=nbcall+1;

Sigma = Sigma+ StepSigma * DirSigma;

Kmatrix = sumKbeta(KH,(Sigma.*Sigma));
[Hstar,cost]= mykernelkmeans(Kmatrix,numclass);
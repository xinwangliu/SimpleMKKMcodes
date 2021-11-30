function [H_normalized,obj]= mykernelkmeans(K,cluster_count)

K = (K+K')/2;
opt.disp = 0;
[H,~] = eigs(K,cluster_count,'LA',opt);
obj = trace(H' * K * H);
H_normalized = H;
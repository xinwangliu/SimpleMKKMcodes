function [H_normalized,obj]= mykernelkmeans(K,cluster_count)

K = (K+K')/2;
opt.disp = 0;
[H,~] = eigs(K,cluster_count,'LA',opt);
% obj = trace(H' * K * H) - trace(K);
obj = trace(H' * K * H);
% H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);
H_normalized = H;
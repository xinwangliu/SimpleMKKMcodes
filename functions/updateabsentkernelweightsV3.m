function [gamma,obj]= updateabsentkernelweightsV3(T,K)

num = size(K,1);
nbkernel = size(K,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U0 = eye(num)-T*T';
a = zeros(nbkernel,1);
for p = 1 : nbkernel
    a(p) = trace( K(:,:,p) * U0);
end
gamma = (1./a)/sum(1./a);
gamma(gamma<eps)=0;
gamma = gamma/sum(gamma);
obj = a'*(gamma.^2);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q = zeros(nbkernel);
% for p = 1:nbkernel
%     Q(p, p) = trace(K(:, :, p)) - trace(T' * K(:,:,p) * T);
% end
% res = mskqpopt(Q, zeros(nbkernel, 1), ones(1, nbkernel), 1, 1, zeros(nbkernel, 1), ones(nbkernel, 1), [], 'minimize echo(0)');
% gamma = res.sol.itr.xx;
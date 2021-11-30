function [H_normalized,theta,objective]= mkkmeans_train(Km,cluster_count)

numker = size(Km, 3);
theta = ones(numker,1)/numker;
K_theta = mycombFun(Km, theta.^2);

opt.disp = 0;
iteration_count = 0;
flag =1;

% %%---
% maxIter = 30;
% res_mean = zeros(4,maxIter);
% res_std = zeros(4,maxIter);

while flag
    iteration_count = iteration_count+1;
    fprintf(1, 'running iteration %d...\n', iteration_count);
    [H, ~] = eigs(K_theta, cluster_count, 'LA', opt);
    % [res_mean(:,iteration_count),res_std(:,iteration_count)] = myNMIACCV2(H,Y,cluster_count);
    %     resH(iteration_count,:) = myNMIACC(H,Y,cluster_count);
    %     Q = zeros(numker);
    %     for m = 1:numker
    %         Q(m, m) = trace(Km(:, :, m)) - trace(H' * Km(:, :, m) * H);
    %     end
    %     res = mskqpopt(Q, zeros(numker, 1), ones(1, numker), 1, 1, zeros(numker, 1), ones(numker, 1), [], 'minimize echo(0)');
    %     theta = res.sol.itr.xx;
    [theta] = updateabsentkernelweightsV3(H,Km);
    K_theta = mycombFun(Km, theta.^2);
    objective(iteration_count) = -trace(H' * K_theta * H) + trace(K_theta);
    if iteration_count>2 && (abs((objective(iteration_count-1)-objective(iteration_count))...
            /(objective(iteration_count-1)))<1e-4|| iteration_count>50)
        flag =0;
    end
    %     if iteration_count>=maxIter
    %         flag =0;
    %     end
    %     if iteration_count>100
    %         flag =0;
    %     end
end
% H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1, cluster_count);
H_normalized = H;
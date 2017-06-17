%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discrete Matrix Completion
% min: l(U'*V, d_{0,1}, ..., d_{s,s+1}) + 0.5*\gamma \|U\|^2_F + 0.5*\gamma
% \|V\|^2_F + 0.5*\alpha \|d-0.5\|^2_F
% s.t. 0 <= d_{0,1}, ..., d_{s,s+1} <= 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, U, V, d] = DMCdec(Data, r, gamma, alpha, Par,X0)

[m, n] = size(Data.M);

if ~exist('X0','var')
    U = rand(r, m);
    V = rand(r, n);
else
    [U,V] = nnmf(X0,r);
    U = U';
    U = U/norm(U);
    V = V/norm(V);
end
d = 0.5 * ones(Data.maxGrade + 1, 1);

numOmega = numel(Data.Omega);
ind = mod(randperm(Par.maxIter), numOmega) + 1;

fprintf('Matrix Decomposition with Discrete Loss function\n');
fprintf('iteration | objective | errorOmega | errorAll \n');

for k = 1:Par.maxIter
    t = Data.Omega(ind(k));
    [i, j] = ind2sub([m, n], t);
    
    stepSize = Par.stepSize / k^Par.stepMode;
    
    tmp = U(:,i)'*V(:, j) - Data.M(t);
    tmp1 = max(tmp - d(Data.M(t)+1), 0); 
    tmp2 = max(-tmp - 1 + d(Data.M(t)), 0);
    
    gU = (tmp1 - tmp2) * V(:, j) + gamma * U(:, i);
    gV = (tmp1 - tmp2) * U(:, i) + gamma * V(:, j);
    
    U(:, i) = U(:, i) - stepSize * gU;
    V(:, j) = V(:, j) - stepSize * gV;
    
    d(Data.M(t)+1) = min(1, max(0, d(Data.M(t)+1) - stepSize * (-tmp1+alpha*(d(Data.M(t)+1)-0.5))));
    d(Data.M(t)) = min(1, max(0, d(Data.M(t)) - stepSize * (tmp2+alpha*(d(Data.M(t))-0.5))));  
    
    % print the middle results
    if mod(k, 10*numOmega) == 1
        % objective
        X = U'*V;
        tmp = X(Data.Omega) - Data.M(Data.Omega);
        f1 = max(tmp-d(Data.M(Data.Omega)+1), 0);
        f2 = max(-tmp - 1 + d(Data.M(Data.Omega)), 0);
        
        f = 0.5*(norm(f1, 'fro')^2 + norm(f2, 'fro')^2 + alpha*norm(d-0.5)^2);
        [~, errOmega] = ComputeError(X, d, Data);
        fprintf('%9d | %9.2f | %10.4f \n', k, f, errOmega);
    end
end

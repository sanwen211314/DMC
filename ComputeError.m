function [err, errOmega] = ComputeError(X, d, Data)

T = numel(Data.M);
TOmega = numel(Data.Omega);

for j = 1:Data.maxGrade
    ind = X>j-1+d(j) & X<=j+d(j+1);
    X(ind) = j;
end
X(X<1) = 1;
X(X>Data.maxGrade) = Data.maxGrade;

err = sqrt(norm(X - Data.M, 'fro')^2 / T);
errOmega = sqrt(norm(X(Data.Omega) - Data.M(Data.Omega), 'fro')^2 / TOmega);
%% readme
%this file is used to run 5 fold cross validation on the matrix completion algorithms
% and test accuracies.
% by Zhouyuan Huo.

clear;
clc;
%% Generate data.
feature getpid
parpool(5);

load('./data');

[m,n] = size(data);

%% INITIALIZATION
nfold = 5;
split = 5;
% ccan = [2:2:10];   % candidate rank set
maxGrade = 5;
minGrade = 1;
Par.maxIter = 100*numel(data)/20 + 1;   % maximal steps
Par.stepSize = 0.01; % initial stepsize
Par.stepMode = 0.1; % stepsize decreases as stepSize = stepSize / k^stepMode; "0" means constant stepsize.
[gamma,alpha] = meshgrid([1e-5,1e-4,1e-3],[1e-2,1e-1,1,1e1,1e2]);
par_r = 5;

parfor t = 1:numel(gamma)
  rmse = zeros(nfold,1);
  mae = zeros(nfold,1);
  Data = struct('M',{},'Omega',{},'maxGrade',{});
%% five fold
  for ifold =1:nfold 
    A = zeros(m,n);
    A1 = zeros(m,n);
    idxob = find(data~=0); % observed index
    idx = randperm(length(idxob)); 
    unit = floor(length(idxob)/split);
    idxA1 = idx(1: unit);
    idxA1 = idxob(idxA1);
    idxA = idx;
    idxA(1:unit) = [];
    idxA = idxob(idxA);
    A(idxA) = 1;
    A1(idxA1) = 1;

    %test if there is column or row has zero entry.
    test = sum(A,1);
    idxtest = find(test==0);
    if length(idxtest)~=0
        A(:,idxtest) = A1(:,idxtest);
        A1(:,idxtest) = 0;
        idxA = find(A==1);
        idxA1 = find(A1==1);
    end
    test = sum(A,2);
    idxtest = find(test==0);
    if length(idxtest)~=0
        A(idxtest,:) = A1(idxtest,:);
        A1(idxtest,:) = 0;
        idxA = find(A==1);
        idxA1 = find(A1==1);
    end
    
    Y = data;
    Y0 = Y;
    
    %% Algo.  DMCdec method and compute normalized matrix
    %average error
    Data(1).M = Y;
    Data(1).Omega = idxA;
    Data(1).maxGrade = maxGrade;
    [X_dec, U_dec, V_dec, d_dec] = DMCdec(Data(1), par_r, gamma(t), alpha(t), Par);
    Xz = r2zDMC(X_dec,d_dec,minGrade,maxGrade,idxA1);
    rmse(ifold) = sqrt(mean((Xz(idxA1)-Y(idxA1)).^2));
    mae(ifold) = mean(abs(Xz(idxA1)-Y(idxA1)));

  end 
  result(t).rmse = rmse;
  result(t).mae  = mae;
  %result(t).X_dec = X_dec;

end

save('Dis_movielens_DMCdec','result');
delete(gcp);

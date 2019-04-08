function [evects,evals] = truepca(dataset)
% [evects,evals] = truepca(dataset)
%
% USUAL WAY TO DO PCA -- find sample covariance and diagonalize
%
% input: dataset 
% note, in dataset, each COLUMN is a datapoint
% the data mean will be subtracted and discarded
% 
% output: evects holds the eigenvectors, one per column
%         evals holds the corresponding eigenvalues
%

[d,N]  = size(dataset);

mm = mean(dataset')';
% dataset = dataset - mm*ones(1,N);

cc = cov(dataset',1);
cc = full(cc);
[cvv,cdd] = eig(cc);
[zz,ii] = sort(diag(cdd));
ii = flipud(ii);
evects = cvv(:,ii);
cdd = diag(cdd);
evals = cdd(ii);



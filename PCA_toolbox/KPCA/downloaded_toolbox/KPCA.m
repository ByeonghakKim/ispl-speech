function [score] = KPCA(feature,kernel,kerneloption)

[eigvect,eigval]=kernelpca(feature,kernel,kerneloption);
if size(feature,1)>=50&& length(eigval)>=50
    eigval = eigval(1:50);
end
dim = length(eigval);
eigvect = eigvect(:,1:dim);
score=kernelpcaproj(feature,feature,eigvect,kernel,kerneloption,1:length(eigval));
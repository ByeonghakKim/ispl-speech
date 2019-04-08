function [tempfeature1 tempfeature2] = do_PCA(tempfeature,kernel,sigma)
T = length(tempfeature);
feature = [];
for t = 1:T
    num_t(t) = size(tempfeature{t},1);
    feature = [feature; tempfeature{t}]; 
end

% feature = cell2mat(feature);

[num,dim] = size(feature);
% if ~strcmp(kernel,'linear')
    if num > 3000
        rn = sort(rnsubset(3000,num),'ascend');
        tempfeature = feature(rn,:);
    else
        tempfeature = feature;
    end
    Distance = Distancesparse(tempfeature,kernel,1);
    mDistance = mean(mean(sqrt(Distance))); % calculate the kernel width
    tempfeature = []; % delete tmpfeature;
% end

switch kernel
    case 'linear'
        kerneloption = [];
    case 'gaussian'
        kerneloption = sigma*mDistance;
end


% if the dataset is larger than 3000, random sample 3000 observations for
% eigvect.

if num > 3000
    rn = sort(rnsubset(3000,num),'ascend');
    tempfeature = feature(rn,:);
else
    tempfeature = feature;
end

[eigvect,eigval]=kernelpca(tempfeature,kernel,kerneloption);

tempfeature1  = dealforall1(eigval,eigvect,kernel,kerneloption,feature,T,num_t);
tempfeature2  = dealforall2(eigval,eigvect,kernel,kerneloption,feature,T,num_t);

function tempfeature1  = dealforall1(eigval,eigvect,kernel,kerneloption,feature,T,num_t)
% only save 90% eigvals
percentage = 0.90;
for m = length(eigval):-1:1
    if sum(eigval(1:m))/sum(eigval)<percentage
        tmp_pca_dim = m+1;
        break;
    end
end
eigval = eigval(1:tmp_pca_dim);


dim = length(eigval);
eigvect = eigvect(:,1:dim);
score=kernelpcaproj(feature,feature,eigvect,kernel,kerneloption,1:length(eigval));

feature = cell(1,T);
for t= 1:T
    feature{t} = sparse(real(score(1:num_t(t),:))); % debug
    score(1:num_t(t),:) = [];
end
tempfeature1 = feature;


function tempfeature2  = dealforall2(eigval,eigvect,kernel,kerneloption,feature,T,num_t)
% only save 90% eigvals
percentage = 0.90;
for m = length(eigval):-1:1
    if sum(eigval(1:m))/sum(eigval)<percentage
        tmp_pca_dim = m+1;
        break;
    end
end
eigval = eigval(1:tmp_pca_dim);

if  length(eigval)>=500
    eigval = eigval(1:500);
end


dim = length(eigval);
eigvect = eigvect(:,1:dim);
score=kernelpcaproj(feature,feature,eigvect,kernel,kerneloption,1:length(eigval));

feature = cell(1,T);
for t= 1:T
    feature{t} = sparse(real(score(1:num_t(t),:))); % debug
    score(1:num_t(t),:) = [];
end

tempfeature2 = feature;

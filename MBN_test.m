function [testfeature] = MBN_test(testdata,model)
% Input:
%       data    -- N x d matrix, N is the number of the data points, d is
%                       the dimension of the data.
%       model   -- Output model of MBN for the prediction stage. If model
%                       is not saved during training, then model is empty.
%
% Output:
%       feature   -- Low-dimensional output of MBN_test.
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contact info: Xiao-Lei Zhang (huoshan6@126.com; xiaolei.zhang9@gmail.com)
% Website: https://sites.google.com/site/zhangxiaolei321/
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

status = system('systeminfo');
if status==0
    os = 'windows';
else 
    status= system('uname -a');
    if status == 0
        os = 'linux';
    else
        os = 'unknown';
    end
end

if ~exist('model','var')
    error('Model does not exist!\n');
elseif isempty(model)
    error('Model is not saved during training!\n');
end

dir = model.dir;
depth = model.param.depth;
similarity = model.param.similarity;
switch os
    case 'linux'
        model_dir = [dir,'/model/']; % if windows OS, use '\', instead of '/'
        feature_dir = [dir,'/feature/'];
        testfeature_dir = [dir,'/testfeature/'];
    case 'windows'
        model_dir = [dir,'\model\']; % if windows OS, use '\', instead of '/'
        feature_dir = [dir,'\feature\'];
        testfeature_dir = [dir,'\testfeature\'];
end

%% MBN prediction
MBN_predict(testdata,ones(size(testdata,1),1),depth,similarity,model_dir,testfeature_dir);  



%% PCA prediction
switch os
    case 'linux'
        PCA_testfeature_file_path = [dir,'/PCA_testfeature/'];
    case'windows'
        PCA_testfeature_file_path = [dir,'\PCA_testfeature\'];
end
layerID = depth;
testfeaturefile = [testfeature_dir,num2str(layerID),'.mat'];
load(testfeaturefile,'testfeature');
PCA_model_file = [model_dir,'pca_model.mat'];
load(PCA_model_file,'PCAmodel');
switch PCAmodel.pca_type
    case 'empca'
        testfeature = testfeature*PCAmodel.pca_model;
    case 'kernelpca'
        [testfeature] = learn_PCA_feature_predict(testfeature,PCAmodel.pca_model);
end

reduced_dim = size(testfeature,2);
switch os
    case 'linux'
        testfeaturedir = [PCA_testfeature_file_path,'reduced_dim',num2str(reduced_dim),'/'];
    case'windows'
        testfeaturedir = [PCA_testfeature_file_path,'reduced_dim',num2str(reduced_dim),'\'];
end
if ~exist(testfeaturedir,'dir')
    mkdir(testfeaturedir);
end
testfeaturefile = [testfeaturedir,num2str(layerID),'.mat'];
save(testfeaturefile,'testfeature','-v7.3');




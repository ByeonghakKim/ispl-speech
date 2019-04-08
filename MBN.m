function [feature,model] = MBN(data,c,param)
% Input:
%       data    -- N x d matrix, N is the number of the data points, d is
%                       the dimension of the data.
%       c       -- number of classes of the input data. If the ground-truth
%                       number of classes is unknown, please make a rough
%                       guess of c. Parameter c is related to the parameter
%                       k at the top nonlinear layer which is 1.5 times
%                       larger than c. Default value = 10.
%       param   -- {'parameter name1',value1,'parameter name2',value2,...} 
%                       is a cell vector specifying the custormized
%                       parameter setting. An example of custormized
%                       parameter setting can be {'kmax', [5000,5000],'V',200,'d',3}
%                       See the following for all possible parameter names:
%                       
%
%       'kmax'    -- The largest matrix that hardware can support.
%                        Default value = [10000,10000].
%       'k1'      -- The parameter k at the bottom layer. If k1 is set,
%                        then kmax is disabled. Default value = 0.5*N.
%       'delta'   -- The speed that parameter decays, i.e. k_{l+1} =
%                        delta*k_l. 'delta' is between [0,1]. Default_value = 0.5;
%       'k'       -- Parameters k of all layers, which is a vector. If k is
%                        user defined, then 'kmax', 'k1', and 'delta' are
%                        disabled.
%       'V'       -- Number of clusterings per layer. Default value = 400.
%       'a'       -- fraction of the randomly selected features for enlarging
%                        the diversity between the clusterings. a is better
%                        to be selected from [0.5, 1]. Default value = 0.5;
%
%       's'       -- the similarity measurement of the bottom layer.
%                        It can be set to 'e' or 'l'.
%                        'e' means Euclidean distance is used; 'l' means
%                        linear kernel is used. Default value = 'e'
%       'm'       -- Decide whether the model will be used for
%                        prediction or not. It can be set to 'yes' or 'no'
%                        'yes' means the model will be saved for the future
%                        prediction job; 'no' means the model wil not be
%                        saved. Defaut value = 'no'.
%       'd'       -- Number of output dimension of PCA at the top layer.
%                        Default value = number of classes (i.e. parameter c).
%       'dir'     -- The directory of model (including MBN model and output PCA model)
%                        where the model is saved. Default value =
%                        [current_path,'tmp_data'].
%
% Output:
%       feature   -- Low-dimensional output of MBN.
%       model     -- Output model of MBN for the prediction stage. If model
%                        is not saved during training, then model is empty. 
%
% NOTE: User may pay attention to the setting of parameter 'delta' (and
% maybe also 'k1'), which determines the size of the network and may affects 
% the performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contact info: Xiao-Lei Zhang (huoshan6@126.com; xiaolei.zhang9@gmail.com)
% Website: https://sites.google.com/site/zhangxiaolei321/
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
current_path=get_platform_filepath();
addpath(genpath(current_path));
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


[num,dim] = size(data);
%% default parameter setting
if ~exist('c','var')
    c = 10;
elseif isempty(c)
    c = 10;
end

kmax = [10000,10000];
max_k1 = floor(kmax(1)*kmax(2)/num);
k1 = min(floor(num*0.5),max_k1);
delta = 0.5;
a = 0.5;
V = 400;
reduced_dim = c;
similarity = 'e';
save_model = 'no'; 
dir = [current_path,'tmp_data'];

%% customized parameter setting
note = zeros(1,10);
try
    len = length(param);
    for i = 1:2:len-1
        tmp_param = param{i};
        switch tmp_param
            case 'kmax'
                if note(2) == 0
                    kmax = param{i+1};
                    max_k1 = floor(kmax(1)*kmax(2)/num);  
                    k1 = min(floor(num*0.5),max_k1);
                    note(1) = 1; 
                else
                    fprintf('Warning: k1 is already set. kmax is disabled.\n');
                end
            case 'k1'
                k1 =  param{i+1};
                note(2);
            case 'delta'
                delta = param{i+1};
                note(3) = 1; 
            case 'k'
                k = param{i+1};                
                note(4) = 1;  
            case 'a'
                a = param{i+1};
                note(5) = 1;
            case 'V'
                V = param{i+1};
                note(6) = 1;
            case 's'
                similarity = param{i+1};
                note(7) = 1;
            case 'm'
                save_model = param{i+1};
                note(8) = 1;
            case 'd'
                reduced_dim = param{i+1};
                note(9) = 1;
            case 'dir'
                dir = param{i+1};
                note(10) = 1;
        end
    end
catch
    
end

if note(4) == 0
    % set k
    k = k1; 
    while 1
       tmp_k = floor(k(end)*delta);
       if tmp_k< 1.5*c
           break;
       end
       k = [k,tmp_k];
    end  
end



%% prepare tmp_data directory
switch os
    case 'linux'
        model_dir = [dir,'/model/']; % if windows OS, use '\', instead of '/'
        feature_dir = [dir,'/feature/'];
    case 'windows'
        model_dir = [dir,'\model\']; % if windows OS, use '\', instead of '/'
        feature_dir = [dir,'\feature\'];
end



%% sparse representation learning
MBN_train(data,ones(num,1),k,V,a,similarity,save_model, model_dir,feature_dir);
   
%% dimensionality reduction by PCA (the top layer of MBN)
switch os
    case 'linux'
        PCA_trainfeature_file_path = [dir,'/PCA_trainfeature/'];
    case'windows'
        PCA_trainfeature_file_path = [dir,'\PCA_trainfeature\'];
end
layerID = length(k);
featurefile = [feature_dir,num2str(layerID),'.mat'];
load(featurefile,'feature');
if num > 3000  % if number of data points is larger than 3000, then use EMPCA
    tmp_reduced_dim = max(min(reduced_dim*3,100),reduced_dim);
    [evec,eval] = empca(feature',tmp_reduced_dim);
    evec = evec(:,1:reduced_dim);
    feature = feature*evec;
    PCAmodel.pca_model = evec;
    PCAmodel.pca_type = 'empca';
else % otherwise, use linear-kernel based kernel-PCA
    [feature,pca_model] = learn_PCA_feature(feature,reduced_dim);
    PCAmodel.pca_model = pca_model;
    PCAmodel.pca_type = 'kernelpca';
end
switch os
    case 'linux'
        featuredir = [PCA_trainfeature_file_path,'reduced_dim',num2str(reduced_dim),'/'];
    case'windows'
        featuredir = [PCA_trainfeature_file_path,'reduced_dim',num2str(reduced_dim),'\'];
end
if ~exist(featuredir,'dir')
    mkdir(featuredir);
end
featurefile = [featuredir,num2str(layerID),'.mat'];
save(featurefile,'feature','-v7.3');
if strcmp(save_model,'yes')
    PCA_model_file = [model_dir,'pca_model.mat'];
    save(PCA_model_file,'PCAmodel','-v7.3');
end

%% the low-dimensional features
layerID = length(k);
switch os
    case 'linux'
        featurefile = [PCA_trainfeature_file_path,'/reduced_dim',num2str(reduced_dim),'/',num2str(layerID),'.mat'];
    case'windows'
        featurefile = [PCA_trainfeature_file_path,'\reduced_dim',num2str(reduced_dim),'\',num2str(layerID),'.mat'];
end
load(featurefile,'feature');


%% output model information
if strcmp(save_model,'yes')
    model.dir = dir;
    model.param.depth = length(k);
    model.param.similarity = similarity;
else
    model = [];
end

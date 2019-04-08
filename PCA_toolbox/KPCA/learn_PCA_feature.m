function [PCA_feature,model] = learn_PCA_feature(feature,dimension,seed_num)

if ~exist('seed_num','var')
    seed_num = 2000;
end

[PCA_feature, model] = do_PCA2(feature,'linear',0,dimension,seed_num);


% 
% PCA_feature = PCA_feature{1};
% model = model{1};

    
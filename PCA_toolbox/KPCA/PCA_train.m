function [] = PCA_train(depth,reduced_dim,learned_trainfeature_file_path,PCA_trainfeature_file_path)

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


for layerID = 0:depth
    switch os
        case 'linux'
            file = [learned_trainfeature_file_path,'/',num2str(layerID),'.mat'];
        case 'windows'
            file = [learned_trainfeature_file_path,'\',num2str(layerID),'.mat'];
    end
    load(file);
    
    [feature, PCAmodel] = learn_PCA_feature(feature,reduced_dim);
    
    switch os
        case 'linux'
            PCA_dir = [PCA_trainfeature_file_path,'/reduced_dim',num2str(reduced_dim),'/'];
        case 'windows'
            PCA_dir = [PCA_trainfeature_file_path,'\reduced_dim',num2str(reduced_dim),'\'];
    end
    
    
    if ~exist(PCA_dir,'dir')
        mkdir(PCA_dir);
    end
    PCA_file = [PCA_dir,num2str(layerID),'.mat'];
    save(PCA_file,'feature','label','PCAmodel','-v7.3');
end

xx = 1;



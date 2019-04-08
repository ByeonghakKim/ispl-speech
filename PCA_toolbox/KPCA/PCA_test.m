function [] = PCA_test(depth,reduced_dim,PCA_trainfeature_file_path,learned_testfeature_file_path,PCA_testfeature_file_path)



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
            file = [learned_testfeature_file_path,'/',num2str(layerID),'.mat'];
        case 'windows'
            file = [learned_testfeature_file_path,'\',num2str(layerID),'.mat'];
    end
    
    load(file);
    
    switch os
        case 'linux'
            modelfile = [PCA_trainfeature_file_path,'/reduced_dim',num2str(reduced_dim),'/',num2str(layerID),'.mat'];
        case 'windows'
            modelfile = [PCA_trainfeature_file_path,'\reduced_dim',num2str(reduced_dim),'\',num2str(layerID),'.mat'];
    end
    
    
    load(modelfile,'PCAmodel');

    
    [testfeature] = learn_PCA_feature_predict(testfeature,PCAmodel);
    
    
    switch os
        case 'linux'
            PCA_dir = [PCA_testfeature_file_path,'/reduced_dim',num2str(reduced_dim),'/'];
        case 'windows'
            PCA_dir = [PCA_testfeature_file_path,'\reduced_dim',num2str(reduced_dim),'\'];
    end
    
    if ~exist(PCA_dir,'dir')
        mkdir(PCA_dir);
    end
    
    switch os
        case 'linux'
            PCA_file = [PCA_dir,'/',num2str(layerID),'.mat'];
        case 'windows'
            PCA_file = [PCA_dir,'\',num2str(layerID),'.mat'];
    end
    
    save(PCA_file,'testfeature','testlabel','PCAmodel','-v7.3');
end






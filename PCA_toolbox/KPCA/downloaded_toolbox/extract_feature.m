function extract_feature
% dir = 'E:\正式研究平台\method\正式方法\ZhaoMKC\feature\';
% 
% %%%%%%%%%%%%测试UCI部分
% flag = 'UCI';
% 
% %[1:12 14:16 20]
% for choose = [ 1:14 16:19]
%     [original_feature,label,mDistance] = getTrainingData(choose,0,flag);
%     [Dis.mDistance,Dis.maxf,Dis.minf,Dis.feature]= Calc_sigma(original_feature);
%     % 提取Gaussian kernel 特征
%     for kernelwidth = -2:2
%         kerneloption = 2^kernelwidth * mDistance;
%         tic;
%         [feature] = KPCA(original_feature,'gaussian',kerneloption);
%         time4feature = toc;
%         feature = sparse(feature);
%         filename = strcat(dir,'UCI_',num2str(choose),'_g_',num2str(kernelwidth),'.mat');
%         save(filename,'feature','label','time4feature');
%     end
%     % 提取polynomial kernel 特征
%     for kernelwidth = 2:4
%         kerneloption = kernelwidth;
%         tic;
%         [feature] = KPCA(original_feature,'poly',kerneloption);
%         time4feature = toc;
%         feature = sparse(feature);
%         filename = strcat(dir,'UCI_',num2str(choose),'_p_',num2str(kernelwidth),'.mat');
%         save(filename,'feature','label','time4feature');
%     end
% 
% 
% end




dir = 'E:\正式研究平台\method\正式方法\ZhaoMKC\feature\';

%%%%%%%%%%%%测试UCI部分
flag = 'UCIadult';

%[1:12 14:16 20]
for choose = [ 1:9]
    [original_feature,label,mDistance] = getTrainingData(choose,0,flag);
    [Dis.mDistance,Dis.maxf,Dis.minf,Dis.feature]= Calc_sigma(original_feature);
    % 提取Gaussian kernel 特征
    for kernelwidth = -2:2
        kerneloption = 2^kernelwidth * mDistance;
        tic;
        [feature] = KPCA(original_feature,'gaussian',kerneloption);
        time4feature = toc;
        feature = sparse(feature);
        filename = strcat(dir,'UCIadult_',num2str(choose),'_g_',num2str(kernelwidth),'.mat');
        save(filename,'feature','label','time4feature');
    end
    % 提取polynomial kernel 特征
    for kernelwidth = 2:4
        kerneloption = kernelwidth;
        tic;
        [feature] = KPCA(original_feature,'poly',kerneloption);
        time4feature = toc;
        feature = sparse(feature);
        filename = strcat(dir,'UCIadult_',num2str(choose),'_p_',num2str(kernelwidth),'.mat');
        save(filename,'feature','label','time4feature');
    end


end
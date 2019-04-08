function [tempfeature1, model] = do_PCA2_predict(feature,model)



tempfeature1  = dealforall1(feature,model);
% tempfeature2  = dealforall2(eigval,eigvect,kernel,kerneloption,feature,tempfeature,T,num_t);

function score  = dealforall1(feature,model)


eigval = model.eigval;
eigvect = model.eigvect;
kernel = model.kernel;
kerneloption = model.kerneloption;
tempfeature = model.tempfeature;
reduced_dim = model.reduced_dim;


% only save 90% eigvals
% percentage = 0.90;
% for m = length(eigval):-1:1
%     if sum(eigval(1:m))/sum(eigval)<percentage
%         tmp_pca_dim = m+1;
%         break;
%     end
% end
% eigval = eigval(1:tmp_pca_dim);

if  length(eigval)>=reduced_dim
    eigval = eigval(1:reduced_dim);
end


dim = length(eigval);
eigvect = eigvect(:,1:dim);


score=distributed_PCA_prediction(tempfeature,feature,eigvect,kernel,kerneloption,1:length(eigval));


% score=kernelpcaproj(tempfeature,feature,eigvect,kernel,kerneloption,1:length(eigval));

% feature = cell(1,T);
% for t= 1:T
%     feature{t} = sparse(real(score(1:num_t(t),:))); % debug
%     score(1:num_t(t),:) = [];
% end
% tempfeature1 = feature;





function [normxsup] = distributed_PCA_prediction(center,xsup,eigvect,kernel,kerneloption,eigvectind) % xsup is to be predicted, center is a small set of training data

%һ�δ�������������
[num,dim] = size(xsup);
% % totalnum = num*dim;
% % blocknum = floor(totalnum/(10*10^8))+1;  %2*10^6��һ�δ�������ݴ�С
% if size(xsup,1) >2000 && size(xsup,1) <=10000
%     blocknum = 8;
% elseif size(xsup,1) > 10000
%     blocknum = 40;
% else
%     blocknum = 1;
% end
% % blocknum = 8;
% if blocknum == 1
%     blocksize = num;
% else
%     blocksize = floor(num/blocknum);
% end


if num >= 10000
    blocksize = 10000;
else
    blocksize = num;
end

% if blocksize < num
%     blocknum = ceil(num/blocksize);
% else
%     blocknum = 1;
% end








% blocksize = 300000;
num = size(xsup,1);
if num>blocksize
    blocknum = floor(num/blocksize);
    tailsize = num-blocksize*blocknum;

    
    nxpca=size(center,1);
%     nxtest = size(xsup,1);
    K=svmkernel(center,kernel,kerneloption); 
    % centering in features spaces
    onepca=(ones(nxpca,nxpca))/nxpca;
    tempA = K*onepca;
%     tempD = 1-onepca;
%     onepca = [];
    
    
    normxsup = zeros(num,size(eigvect,2));
    
    count = 0;
    if tailsize ~= 0
        count = count + 1;
%         tmpcell{count,1} = xsup(1:tailsize,:)*center';
        xtest = xsup(1:tailsize,:);
        nxtest=size(xtest,1);
        Ktest=svmkernel(xtest,kernel,kerneloption,center); 
        onetest=(ones(nxtest,nxpca))/nxpca;
        tempB = -onetest*K + onetest*tempA;
        Kt = Ktest-Ktest*onepca +tempB;
        normxsup(1:tailsize,:)=full(Kt*eigvect(:,eigvectind));
        
%         normxsup(1:tailsize,:) = kernelpcaproj(center,xsup(1:tailsize,:),eigvect,kernel,kerneloption,eigvectind);
%         normxsup(1:tailsize,:) = xsup(1:tailsize,:)*center';
    end
    

    
    if tailsize ~= 0
        nxtest=blocksize;
        onetest=(ones(nxtest,nxpca))/nxpca;
        tempB = -onetest*K + onetest*tempA; 
        for i = 1:blocknum 
%         count = count + 1;
            start = tailsize+(i-1)*blocksize+1;
            End = start+blocksize-1;
        
            xtest = xsup(start:End,:);
%             nxtest=size(xtest,1);
            Ktest=svmkernel(xtest,kernel,kerneloption,center); 
%             onetest=(ones(nxtest,nxpca))/nxpca;
%             tempB = -onetest*K + onetest*tempA;
            Kt = Ktest-Ktest*onepca +tempB;
            normxsup(start:End,:)=full(Kt*eigvect(:,eigvectind));
            
            
%             tmpcell{i+1,1} = xsup(start:End,:)*center';
%             normxsup(start:End,:) = kernelpcaproj(center,xsup(start:End,:),eigvect,kernel,kerneloption,eigvectind);
            fprintf('blocknum %d\n',i);
%             normxsup(start:End,:) = xsup(start:End,:)*center';
        end
    else
        nxtest=blocksize;
        onetest=(ones(nxtest,nxpca))/nxpca;
        tempB = -onetest*K + onetest*tempA; 
        for i = 1:blocknum 
%         count = count + 1;
            start = tailsize+(i-1)*blocksize+1;
            End = start+blocksize-1;
            
            xtest = xsup(start:End,:);
%             nxtest=size(xtest,1);
            Ktest=svmkernel(xtest,kernel,kerneloption,center); 
%             onetest=(ones(nxtest,nxpca))/nxpca;
%             tempB = -onetest*K + onetest*tempA;
            Kt = Ktest-Ktest*onepca +tempB;
            normxsup(start:End,:)=full(Kt*eigvect(:,eigvectind));
            
            
%             tmpcell{i,1} = xsup(start:End,:)*center';
%             normxsup(start:End,:) = kernelpcaproj(center,xsup(start:End,:),eigvect,kernel,kerneloption,eigvectind);
            fprintf('blocknum %d\n',i);
%             normxsup(start:End,:) = xsup(start:End,:)*center';


        end
    end
    

%     normxsup = sparse(num,size(center,1));

%     l = length(tmpcell);
%     normxsup = [];
%     for i = 1:l
%         normxsup = [normxsup;tmpcell{i}];
%         tmpcell{i} = [];
%     end
    
%     normxsup = cell2mat(tmpcell);
else
%     normxsup = xsup*center';
    normxsup = kernelpcaproj(center,xsup,eigvect,kernel,kerneloption,eigvectind);
end
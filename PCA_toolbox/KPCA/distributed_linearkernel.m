function [normxsup] = distributed_linearkernel(xsup,center)

%一次处理三百万数据
[num,dim] = size(xsup);
% totalnum = num*dim;
% blocknum = floor(totalnum/(10*10^8))+1;  %2*10^6是一次处理的数据大小
if size(xsup,1) >2000 && size(xsup,1) <=10000
    blocknum = 8;
elseif size(xsup,1) > 10000
    blocknum = 40;
else
    blocknum = 1;
end

% blocknum = 8;
if blocknum == 1
    blocksize = num;
else
    blocksize = floor(num/blocknum);
end



% blocksize = 300000;
num = size(xsup,1);
if num>blocksize
    blocknum = floor(num/blocksize);
    tailsize = num-blocksize*blocknum;

    if tailsize == 0
        tmpcell = cell(blocknum,1);
    else
        tmpcell = cell(blocknum+1,1);
    end
    
    normxsup = sparse(num,size(center,1));
    
    count = 0;
    if tailsize ~= 0
        count = count + 1;
        tmpcell{count,1} = xsup(1:tailsize,:)*center';
%         normxsup(1:tailsize,:) = xsup(1:tailsize,:)*center';
    end
    if tailsize ~= 0
        for i = 1:blocknum 
%         count = count + 1;
            start = tailsize+(i-1)*blocksize+1;
            End = start+blocksize-1;
            tmpcell{i+1,1} = xsup(start:End,:)*center';
%             normxsup(start:End,:) = xsup(start:End,:)*center';
        end
    else
        for i = 1:blocknum 
%         count = count + 1;
            start = tailsize+(i-1)*blocksize+1;
            End = start+blocksize-1;
            tmpcell{i,1} = xsup(start:End,:)*center';
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
    
    normxsup = cell2mat(tmpcell);
else
    normxsup = xsup*center';
end
Use MATLAB command 'matlabpool' (or 'parpool' for the latest versions of MATLAB) to start parallel computing. 
Run 'demo.m' for an example. The main programs of MBN for dimensionality reduction are 'MBN.m' and 'MBN_test.m'.


Note that (i) Parameter c in MBN.m indicates the number of true classes of data, default is 10. Parameter c usually should be given by user.
(ii) If the model will be used for test, then {'m','yes'} must be specified.


All parameters are as follows. We recommend users to set 'kmax' as large as possible, set 'm' to 'yes' if the MBN model will be used for prediction, and set 'c' to the ground-truth classes of data.

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
%                       parameter setting can be {'kmax', 5000,'V',200,'d',3}
%                       See the following for all possible parameter names:
%                       
%
%       'kmax'    -- The largest k (at the bottom layer) that hardware can support.
%                        Default value = 5000.
%       'k'       -- Parameters k of all layers, which is a vector.
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
%                        When r=0.5, default value = floor(log2(c))+2; when r = 0,
%                        default value = number of classes (i.e. parameter c).
%       'dir'     -- The directory of model (including MBN model and output PCA model)
%                        where the model is saved. Default value =
%                        [current_path,'tmp_data'].
%
% Output:
%       feature   -- Low-dimensional output of MBN.
%       model     -- Output model of MBN for the prediction stage. If model
%                        is not saved during training, then model is empty.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Contact info: Xiao-Lei Zhang (huoshan6@126.com; xiaolei.zhang9@gmail.com)
% Website: https://sites.google.com/site/zhangxiaolei321/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

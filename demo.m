clear; clc;

%% load data set
load('1.AMLALL.mat','trainfeature','trainlabel','testfeature','testlabel');


%% MBN dimensionality reduction
c = length(unique(trainlabel));
[trainfeature,model] = MBN(trainfeature,c,{'m','yes','d',2}); % Parameter c indicates the number of true classes of data, default is 10. 
                                                        % User may add custormized parameters in the form of, e.g. {'d',2,'V', 2000,'kmax',5000}, 
                                                        % but if the model will be used for test, then {'m','yes'} must be specified.
[testfeature] = MBN_test(testfeature,model);


%% Visualize results
figure; 
feature = [trainfeature;testfeature];
x1max = max(feature(:,1));
x1min = min(feature(:,1));
x2max = max(feature(:,2));
x2min = min(feature(:,2));
subplot(1,2,1);
plot(trainfeature(trainlabel==1,1),trainfeature(trainlabel==1,2),'b+');hold on;
plot(trainfeature(trainlabel==2,1),trainfeature(trainlabel==2,2),'ro');
title('Train');
axis([x1min x1max x2min x2max]);
axis equal;

subplot(1,2,2);
plot(testfeature(testlabel==1,1),testfeature(testlabel==1,2),'b+');hold on;
plot(testfeature(testlabel==2,1),testfeature(testlabel==2,2),'ro');
title('Test');
axis([x1min x1max x2min x2max]);
axis equal;



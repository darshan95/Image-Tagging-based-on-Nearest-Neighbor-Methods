clear

randSeed = 1;
randn('state' , randSeed);
rand('state' , randSeed);

%load('corel5k.mat','NN','Y')

load('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/feat2/train/trainFeatures.mat','projTrainFtrs');
load('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/feat2/test/testFeatures.mat','projTestFtrs');
test_annot = vec_read('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/corel5k_test_annot.hvecs');
train_annot = vec_read('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/corel5k_train_annot.hvecs');
train_features = compute_train_features(projTrainFtrs);
%dist = pdist2(train_features,train_features,'euclidean');
load('dist_euclid.mat','dist');
%disp('size dist');
%size(dist)
%save('dist_euclid.mat','dist')
max_dist = max(max(dist));
min_dist = min(min(dist));
dist = (dist - min_dist)/(max_dist - min_dist);


k=100;
NN = zeros(k,size(dist,2));
%compute ND and NN
ND = zeros(1,size(NN,1),size(NN,2));
for i=1:size(dist,2)
    s = dist(:,i);
    size(s);
    [sorted sindex] = sort(s);
    NN(:,i) = sindex(2:k+1);
    ND(1,:,i) = sorted(2:k+1);
end
%for i=1:size(train_features,1)
 %   for j=1:size(NN,1)
  %      ND(1,j,i) = dist(i,NN(j,i));
  %  end
%end
size(NN);
%max_nd = max(max(ND));
%min_nd = min(min(ND));

%ND = (ND - min_nd)/(max_nd - min_nd);

[model ll] = tagprop_learn(NN,ND,double(train_annot'),'type','rank','sigmoids',false);

%compute NN_test and ND_test
dist2 = pdist2(train_features,projTestFtrs,'euclidean');
max_dist2 = max(max(dist2));
min_dist2 = min(min(dist2));
dist2 = (dist2 - min_dist2)/(max_dist2 - min_dist2);
NN_test = zeros(k,size(dist2,2));
ND_test = zeros(1,k,size(dist2,2));
for i=1:size(dist2,2)
    s = dist2(:,i);
    size(s);
    [sorted sindex] = sort(s);
    NN_test(:,i) = sindex(2:k+1);
    ND_test(1,:,i) = sorted(2:k+1);
end

Y_out = tagprop_predict(NN_test,ND_test,model);
perf = computePerf(test_annot,Y_out',5);
perf
%max_nd = max(max(ND));
%min_nd = min(min(ND));

%ND = (ND - min_nd)/(max_nd - min_nd);
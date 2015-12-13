clear

randSeed = 1;
randn('state' , randSeed);
rand('state' , randSeed);

corel_path = '/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/'
feats1 = struct('feats', {{'DenseHue', 'chisq'}, {'DenseHueV3H1', 'chisq'}, {'DenseSift', 'chisq'}, {'DenseSiftV3H1', 'chisq'}, {'Gist', 'euclidean'}, {'HarrisHue', 'chisq'}, {'HarrisHueV3H1' 'chisq'}, {'HarrisSift', 'chisq'}, {'HarrisSiftV3H1', 'chisq'}, {'Hsv', 'L1'}, {'HsvV3H1', 'L1'}, {'Lab', 'L1'}, {'LabV3H1', 'L1'}, {'Rgb', 'L1'}, {'RgbV3H1', 'L1'}});
pd_flag = 1;
load('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/feat2/train/trainFeatures.mat','projTrainFtrs');
load('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/feat2/test/testFeatures.mat','projTestFtrs');
test_annot = vec_read('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/corel5k_test_annot.hvecs');
train_annot = vec_read('/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/corel5k_train_annot.hvecs');

train_features = zeros(size(projTrainFtrs,1),size(projTrainFtrs,2));
fileID = fopen('randpick.txt','r');
formatSpec = '%d';
A = fscanf(fileID,formatSpec);
for i=1:size(train_features,1)
    train_features(A(i),:) = projTrainFtrs(i,:);
end

dist_nn = zeros(15,size(train_features,1),size(train_features,1));
dist2_nn = zeros(15,size(train_features,1),size(projTestFtrs,1));
for i=1:15
    fd = feats1(i).feats;
    feat = cell2mat(fd(1));
    metric = cell2mat(fd(2));
    train = dir(strcat(corel_path, 'corel5k_train_', feat, '.*vec*'));
    test = dir(strcat(corel_path, 'corel5k_test_', feat, '.*vec*'));
    train.name
    test.name
    %a = strcat(corel_path, 'corel5k_train_', feat, '.*vec*')
    %train = vec_read()
    train = vec_read(strcat(corel_path,train.name));
    test = vec_read(strcat(corel_path,test.name));
    %{   
    if pd_flag
        m = size(train, 1);
        n = size(test, 1);
        pairwise_distances = zeros(m, m);
        dist2 = zeros(m, n);
        pd_flag = 0;
    end
    %}
    %if strcmp(metric, 'chisq')
        %temp = pdist2(train, train, 'cityblock');
     %   temp = chisq(train, train);
      %  temp2 = chisq(train, test);
    %elseif strcmp(metric, 'kld')
        %temp = pdist2(train, train, 'cityblock');
     %   temp = chisq(train, train);
        %temp = KLD(train, test);
    %else
    temp = pdist2(train, train, metric);
    save(strcat('train_',feat,'.mat'),'temp');
    temp2 = pdist2(train, test, metric);
    save(strcat('test_',feat,'.mat'),'temp2');
    %end
    temp(isnan(temp))=0;
    mn = min(min(temp));
    mx = max(max(temp));
    norm_distances = (temp -  mn) / (mx - mn);
    dist_nn(i,:,:) = norm_distances;
    %pairwise_distances = pairwise_distances + norm_distances;
    
    temp2(isnan(temp2))=0;
    mn = min(min(temp2));
    mx = max(max(temp2));
    temp2 = (temp2 -  mn) / (mx - mn);
    dist2_nn(i,:,:) = temp2;
end
%pairwise_distances = pairwise_distances ./ 15;
%mn = min(min(pairwise_distances));
%mx = max(max(pairwise_distances));
%norm_distances = (pairwise_distances -  mn) / (mx - mn);
%dist = norm_distances;
save('dist_nn.mat','dist_nn');
%}
%{
load('feat1_dist_tagprop_new.mat','dist');
disp('dist');
size(dist)
%}

%{
dist2 = dist2 ./ 15;
mn = min(min(dist2));
mx = max(max(dist2));
dist2 = (dist2 -  mn) / (mx - mn);
disp('dist2');
size(dist2)
%}
save('dist2_nn.mat','dist2_nn');
%{
load('feat1_dist2_tagprop_new.mat','dist2');
disp('dist2');
size(dist2)
%}
%Computing 4500 x 4500 matrix by adding all 15 distances
dist = squeeze(sum(dist_nn,1)/size(dist_nn,1));
mn = min(min(dist));
mx = max(max(dist));
dist = (dist -  mn) / (mx - mn);

%Computing 4500 x 499 matrix by adding all 15 distances
dist2 = squeeze(sum(dist2_nn,1)/size(dist2_nn,1));
mn = min(min(dist2));
mx = max(max(dist2));
dist2 = (dist2 -  mn) / (mx - mn);
disp('dist2');
disp(size(dist2));

k=100;
NN = zeros(k,size(dist,2));
%compute NN
ND = zeros(15,k,size(NN,2));
for i=1:size(dist,2)
    s = dist(:,i);
    size(s);
    [sorted sindex] = sort(s);
    NN(:,i) = sindex(2:k+1);
    ND(:,:,i) = dist_nn(:,sindex(2:k+1),i);
end
disp('NN');
size(NN)
disp('ND');
size(ND)
[model ll] = tagprop_learn(NN,ND,double(train_annot'),'type','dist','sigmoids',false);

NN_test = zeros(k,size(dist2,2));
ND_test = zeros(15,k,size(dist2,2));
for i=1:size(dist2,2)
    s = dist2(:,i);
    size(s);
    [sorted sindex] = sort(s);
    NN_test(:,i) = sindex(1:k);
    ND_test(:,:,i) = dist2_nn(:,sindex(1:k),i);
end
disp('NN_test');
disp(size(NN_test));
disp('ND_test');
disp(size(ND_test));

Y_out = tagprop_predict(NN_test,ND_test,model);
perf = computePerf(test_annot,Y_out',5);
perf

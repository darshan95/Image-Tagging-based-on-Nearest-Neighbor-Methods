clear

randSeed = 1;
randn('state' , randSeed);
rand('state' , randSeed);
corel_path = '/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/feat2/'
load(strcat(corel_path,'train/','trainFeatures.mat'),'projTrainFtrs');
load(strcat(corel_path,'test/','testFeatures.mat'),'projTestFtrs');
new_train_features = zeros(size(projTrainFtrs,1),size(projTrainFtrs,2));
fileID = fopen(strcat(corel_path,'train/','randpick.txt'),'r');
formatSpec = '%d';
A = fscanf(fileID,formatSpec);
for i=1:size(projTrainFtrs,1)
    new_train_features(A(i),:) = projTrainFtrs(i,:);
end
corel_path1 = '/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/'
train_annot = vec_read(strcat(corel_path1,'corel5k_train_annot.hvecs'));
test_annot = vec_read(strcat(corel_path1,'corel5k_test_annot.hvecs'));

dist_jec_feat2 = pdist2(new_train_features,projTestFtrs,'euclidean');
mn =min(min(dist_jec_feat2));
mx = max(max(dist_jec_feat2));
dist_jec_feat2 = (dist_jec_feat2 - mn)/(mx - mn);
perf = JEC(dist_jec_feat2, train_annot, test_annot, 5);
perf
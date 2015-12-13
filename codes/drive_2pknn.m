train_features = load('../../data/feat2/train/trainFeatures.mat');
test_features = load('../../data/feat2/test/testFeatures.mat');
train_features = train_features(1).projTrainFtrs;
test_features = test_features(1).projTestFtrs;
trainAnnotations = vec_read('../../data/corel5k.20091111/corel5k_train_annot.hvecs');
testAnnotations = vec_read('../../data/corel5k.20091111/corel5k_test_annot.hvecs');
K1=5;
w=15;
annotLabels=5;
final_distance = load('distance_vI_vJ_feat2.mat');
final_distance = final_distance.final_dist_matrix;
twopassknn(final_distance, trainAnnotations, testAnnotations, K1, w, annotLabels);
finans=zeros(30,4);
%for w=1:30
%	finans(w,:) = twopassknn(final_distance, trainAnnotations, testAnnotations, K1, w, annotLabels);
%end
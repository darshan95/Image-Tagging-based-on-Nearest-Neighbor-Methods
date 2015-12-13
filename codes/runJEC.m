clear

randSeed = 1;
randn('state' , randSeed);
rand('state' , randSeed);

corel_path = '/home/darshan/Dropbox/sem-7/machine_learning/Assignment1 ML 2k15/data/corel5k.20091111/'
feats1 = struct('feats', {{'DenseHue', 'cityblock'}, {'DenseHueV3H1', 'cityblock'}, {'DenseSift', 'chisq'}, {'DenseSiftV3H1', 'chisq'}, {'Gist', 'euclidean'}, {'HarrisHue', 'cityblock'}, {'HarrisHueV3H1' 'cityblock'}, {'HarrisSift', 'chisq'}, {'HarrisSiftV3H1', 'chisq'}, {'Hsv', 'cityblock'}, {'HsvV3H1', 'cityblock'}, {'Lab', 'kld'}, {'LabV3H1', 'kld'}, {'Rgb', 'cityblock'}, {'RgbV3H1', 'cityblock'}});
pd_flag = 1;
train_annot = vec_read(strcat(corel_path,'corel5k_train_annot.hvecs'));
test_annot = vec_read(strcat(corel_path,'corel5k_test_annot.hvecs'));

%trainAnnotations = vec_read('corel5k_train_annot.hvecs');
%testAnnotations = vec_read('corel5k_test_annot.hvecs');

for i=1:15
    fd = feats1(i).feats;
    feat = cell2mat(fd(1));
    metric = cell2mat(fd(2));
    train = dir(strcat(corel_path, 'corel5k_train_', feat, '.*vec*'));
    test = dir(strcat(corel_path, 'corel5k_test_', feat, '.*vec*'));
    train.name
    train = vec_read(strcat(corel_path,train.name));
    test = vec_read(strcat(corel_path,test.name));
       
    if pd_flag
        m = size(train, 1);
        n = size(test, 1);
        pairwise_distances = zeros(m, n);
        pd_flag = 0;
    end
    if strcmp(metric, 'chisq')
        temp = pdist2(train, test, 'cityblock');
        %temp = chisq(train, test);
    elseif strcmp(metric, 'kld')
        temp = pdist2(train, test, 'cityblock');
        %temp = chisq(train, test);
        %temp = KLD(train, test);
    else
        temp = pdist2(train, test, metric);
    end
    temp(isnan(temp))=0;
    mn = min(min(temp));
    mx = max(max(temp));
    norm_distances = (temp -  mn) / (mx - mn);
    pairwise_distances = pairwise_distances + norm_distances;
end

pairwise_distances = pairwise_distances ./ 15;
mn = min(min(pairwise_distances));
mx = max(max(pairwise_distances));
norm_distances = (pairwise_distances -  mn) / (mx - mn);

%trainAnnotations = load('trainAnnotRed');
%trainAnnotations = trainAnnotations.trainAnnotRed;
perf = JEC(norm_distances, train_annot, test_annot, 5);

%matPath = '/home/irshad/Course_Work/ML/Assignment1/Assignment1_ML_2k15/shrinkPredicted/';
%save(strcat(matPath, 'JECF1FR'), 'P');
perf
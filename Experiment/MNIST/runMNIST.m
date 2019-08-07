dt = 3e-3;
T =250;
scheme ={'fixed'};
load('mnist_double.mat')
% load('mnist_randperm.mat')
% train_x = train_x(train_rand(1:6000),:);
% train_labels = train_labels(train_rand(1:6000));
% test_x = test_x(test_rand(1:1000),:);
% test_labels = test_labels(test_rand(1:1000));

Tau = [25e-3];
Mul = [1];
MaxEpoch = 100;
result_grid = cell(length(scheme),1);
for i = length(scheme)
    result_grid{i} = zeros(length(Tau),length(Mul));
end
for ischeme = 1:length(scheme)
    nAfferents = size(train_x,2);
    Data.mixture_mode = 1;
    Data.Labels_name = {'0','1','2','3','4','5','6','7','8','9'};
    Data.ptn = cell(size(train_x,1),nAfferents);
    Data.Labels = train_labels;
    for i = 1:size(train_x,1)
        spike_trains = generate_spikes(scheme{ischeme},train_x(i,:),0.5,784,T);
        %
            Data.ptn(i,:) = st2ptn(spike_trains,dt);
            Data.Tmax(i,1) = dt * T;
            disp([scheme{ischeme},' train:',num2str(i)]);
        %end
    end

    
    nAfferents = size(test_x,2);
    TestData.mixture_mode = 1;
    TestData.Labels_name = {'0','1','2','3','4','5','6','7','8','9'};
    TestData.ptn = cell(size(test_x,1),nAfferents);
    TestData.Labels = test_labels;
    for i = 1:size(test_x,1)
      spike_tests = generate_spikes(scheme{ischeme},test_x(i,:),0.5,784,T);
        %
            TestData.ptn(i,:) = st2ptn(spike_tests,dt);
            TestData.Tmax(i,1) = dt * T;
            disp([scheme{ischeme},' test:',num2str(i)]);
        %end
    end
    Accuracy = [];
    Max_Acc = [];
    for irun  =1 :15
        for itau = 1:length(Tau)
            for imul = 1:length(Mul)
                path = ['/home/gpj/gpj/gpj/STCA-DSNN/Data/Structure/MNIST/Structure_grid_dropout_',scheme{ischeme},'_',num2str(irun),'.mat'];
    %             path = 0;
                cur_tau = [Tau(itau),Tau(itau),Tau(itau)*Mul(imul)];
                Structure = MultilayerTempotronTr_v2(Data,TestData,1,MaxEpoch,[800],1,0,0,path,cur_tau,0.5)
%                 result_grid{ischeme}(itau,imul) = Structure.best_result.accuracy
            end
        end
        Accuracy = [Accuracy;Structure.Errlog{4,1}];
        Max_Acc = [Max_Acc;max(Structure.Errlog{4,1})];
    end
end
save(['../../Data/Data/MNIST/All_Accuracies.mat'],'Accuracy','Max_Acc');
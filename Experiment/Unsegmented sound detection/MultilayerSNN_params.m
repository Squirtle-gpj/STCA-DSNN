

% load('trainData4.mat');
% Data = trainData;
% ValData  = trainData;
% load('trainData1-v3.mat');
% ValData2 = trainData1;
% ValData2.Labels = ValData2.actNumList;

load('RWCP_10_IJCAI_128.mat');
% load('RWCP-trainData8-2.mat');
Data = trainData5;
ValData  = TrainData;
% load('aggregate-label/RWCP-trainData6-4-400-99-v1.mat');
% load('RWCP-TestSpkData-8-2.mat');
ValData2 = TestData;
% ValData2.Labels = ValData2.actNumList;


%params
dt = 3e-3;
tau_m = 30e-3;
dropout = 1;
nCls = 10;
% experiment2: the curvergence curve
% t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev5.ptx','t_v_alter_multispikev5.cu');
% desired = 5;
% maxEpoch = 60;
% Layers = [];
Output_neurons = get_output_neurons(nCls,desired);
nGroups = 1;
maxEpoch = 100;
% Layers = [];
startEpoch = 1;
existweights = 0;
curStructure =  0;
path = 0;

% plug-in components

%Experiment1: observing the output spike timing with the number of input samples increasing.
% s = [20,70,120,170,220,270,320,370,420,470];
% % sp = splice_ptn(trainData1.ptn,s,0.2);
% sp = splice_ptn(testData.ptn,s,0.1);
% spliced_Tmax =  get_Tmax2(sp);
% spliced_ptn = ptn2df(sp,dt); 
% Ninput_samples = 0;
% output_spike_timing = [];
% %visulization command:
%output_spike_timing = gather(output_spike_timing);
%scatter(output_spike_timing(:,1),output_spike_timing(:,2),1,output_spike_timing(:,3),'filled');
%export_fig('output_spike_timing.png','-r300','-transparent');



run MultilayerSNNTr_v3_script.m
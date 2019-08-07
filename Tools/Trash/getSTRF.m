 path = 'D:\代码包\Spatio-temporal neuron model\wav';
 savepath = 'D:\代码包\Spatio-temporal neuron model\cutwav2';
 Structure_path = 'D:\代码包\Spatio-temporal neuron model\M8_Structure.mat';
 load(Structure_path);
 fs = 16000;
 durationCut = 5;

audiocut(path,savepath,fs,durationCut);
response = get_output(Structure,savepath,fs);
auditoryneuron(response,savepath,Structure.nCls*Structure.nGroups);
function audiocut(path,savepath,fs,durationCut)
% path = 'D:\代码包\Spatio-temporal neuron model\wav';
% savepath = 'D:\代码包\Spatio-temporal neuron model\cutwav';
    filelist = dir([path,'\*wav']);
% filelist(1:2) = [];

for ifile = 1:length(filelist)
    disp(filelist(ifile).name)
     [wavtemp, fs_wav] = audioread([path,'\',filelist(ifile).name]) ;
    if size(wavtemp,2)==2
        wavtemp = wavtemp(:,1) + wavtemp(:,2);
        wavtemp = wavtemp/2;
    end
    wavtemp = resample(wavtemp, fs, fs_wav) ;
    wavtemp = wavtemp/max(abs(wavtemp));
    wavtemp(find(wavtemp==0))=[];
    if length(wavtemp)>durationCut*fs
        wavtemp = wavtemp(1:durationCut*fs);
    end
    audiowrite([savepath,'\',filelist(ifile).name],wavtemp,fs);
end
end
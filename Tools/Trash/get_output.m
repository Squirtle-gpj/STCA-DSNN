function response = get_output(Structure,path,fs)
%------parameters----------------------------------
[nAfferents,nOutputs] = size(Structure.AllWeights);
rank = length(Structure.cds);
nGroups = nOutputs/Structure.nCls;
numKernels = nAfferents/rank;

lowFreq = 20;%核函数最低的频率
kernels = get_kernel(fs,numKernels,lowFreq);

filelist = dir([path,'\*wav']);

kernels(end,:) = [];

nptn = 1
for ifile = 1:length(filelist)
    disp(filelist(ifile).name);
     [wavtemp, fs_wav] = audioread([path,'\',filelist(ifile).name]) ;
%     if size(wavtemp,2)==2
%         wavtemp = wavtemp(:,1) + wavtemp(:,2);
%         wavtemp = wavtemp/2;
%     end
%     wavtemp = resample(wavtemp, fs, fs_wav) ;
%     wavtemp = wavtemp/max(abs(wavtemp));
    ws = temporalMP(wavtemp,kernels,true,Structure.rate*(length(wavtemp)/fs_wav));
    ptn(nptn,:) = get_ptn_frm_ws(ws,fs,Structure.cds,Structure.codingmethod);
    nptn = nptn + 1;
end
nptn = nptn - 1;
Tmax = get_Tmax2(ptn);

%------------------initial network----------------------------
Structure.AllWeights = {0,Structure.AllWeights};
Structure.Layers = [nAfferents,nOutputs];
tau_m = 30e-3;
tau_s = tau_m/4;
beta = tau_m/tau_s;
V0 = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
dt = 3e-3;
gdt = gpuArray(double(dt));
threshold = 1;
gthreshold = gpuArray(double(threshold));
gdecay1 = gpuArray(double(exp(-(dt/tau_m))));
gdecay2 = gpuArray(double(exp(-(dt/tau_s))));
ptn = ptn2df(ptn,dt);

nLayers = length(Structure.Layers);
nThreadsperBlock = 64;
T_start = floor((500e-3)/dt);
% Structure.AllWeights = {Structure.AllWeights};

input = cell(1,nLayers);
output = cell(1,nLayers);
u = cell(1,nLayers);
sumLayers = sum(Structure.Layers);
compute_u_kernel = parallel.gpu.CUDAKernel('compute_u_kernel.ptx','compute_u_kernel.cu');
compute_u_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
response = cell(1,nOutputs);
for pp = 1:nptn
    T_size = ceil((Tmax(pp)+2*tau_m)/dt);
    gT_size = gpuArray(int32(T_size));
    
   nstart = 1;
   for ilayer = 1:nLayers
       %-------------for input,output,u
          if ilayer == 1
              output{ilayer} = zeros(T_size,Structure.Layers(ilayer));
              for i = 1:length(ptn{pp})
                    output{ilayer}(ptn{pp}(i,2),ptn{pp}(i,1)) = 1;
              end
               output{ilayer} = gpuArray(double(output{ilayer}));%T_size * Ncurlayer
               continue;
          end
          input{ilayer} = output{ilayer-1}*Structure.AllWeights{ilayer}*V0;
          output{ilayer} = gpuArray(zeros(T_size,Structure.Layers(ilayer),'double'));
          u{ilayer} = gpuArray(zeros(T_size,Structure.Layers(ilayer),'double'));
          compute_u_kernel.GridSize = [ceil(Structure.Layers(ilayer)/nThreadsperBlock),1,1];
          [output{ilayer},u{ilayer}] = feval(compute_u_kernel,u{ilayer},output{ilayer},input{ilayer},gpuArray(int32(T_size)),...
          gpuArray(int32(Structure.Layers(ilayer))),gdecay1,gdecay2,gthreshold);
%                    output{ilayer} = double(output{ilayer});
   end
   for ioutput = 1:nOutputs
       response{ioutput}{pp,1} = {gather(find(output{nLayers}(:,ioutput))*dt*1000)};
   end
end

end
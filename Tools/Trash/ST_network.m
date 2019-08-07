% 11/21 write 'compute_u_kernel.cu' for computing the voltage of each neuron on  GPU.
% 11/22 the forward computation.
% 12/3 write the compute_output_kernel.cu, compute_hidden_kernel.cu
% and compute_u_kernel_without_threshold.cu
tic
nGroups = 1;
startEpoch = 1;
Structure.Layers =300;
%----------params----------------------------
maxEpoch = 30;
existweights = 0;
[nPtns,nAfferents] = size(Data.ptn);
nCls = length(Data.Labels_name);
Tmax = Data.Tmax;
tau_m = 30e-3;
% gtau = gpuArray(double(tau));
tau_s = tau_m/4;
beta = tau_m/tau_s;
V0 = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
dt = 3e-3;
gdt = gpuArray(double(dt));
threshold = 1;
gthreshold = gpuArray(double(threshold));
gdecay1 = gpuArray(double(exp(-(dt/tau_m))));
gdecay2 = gpuArray(double(exp(-(dt/tau_s))));
ptn = ptn2df(Data.ptn,dt);

ClassLabels = Data.Labels;
Labels_name = Data.Labels_name;
nOutputs = nCls*nGroups;
Structure.Layers = [nAfferents,Structure.Layers,nOutputs]
% Structure.Layers = gpuArray(int8(Structure.Layers));
nLayers = length(Structure.Layers);
nThreadsperBlock = 64;


T_start = floor((500e-3)/dt);


%------------------inital----------------------------------------------------
Structure.AllWeights = cell(1,nLayers); 
DW = cell(1,nLayers); 
input = cell(1,nLayers);
output = cell(1,nLayers);
%b = cell(1,nLayers);
u = cell(1,nLayers);
sumLayers = sum(Structure.Layers);
Structure.Acclog = cell(nCls+2,3);
if existweights ~= 1
    for ilayer = 2:nLayers
        Structure.AllWeights{ilayer} = gpuArray(double(...
            1e-3*randn(Structure.Layers(ilayer-1),Structure.Layers(ilayer))));
    end
end

compute_u_kernel = parallel.gpu.CUDAKernel('compute_u_kernel.ptx','compute_u_kernel.cu');
compute_u_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];

for iepoch = startEpoch:maxEpoch
%       RandomSqc=randperm(nPtns);  % random perm the training sequence.
      RandomSqc=1:nPtns;
      Classinfo = ClassLabels(RandomSqc);
      TTmax = Tmax(RandomSqc);
      result = zeros(nCls+2,6);
      for  pp = 1:nPtns
          disp(pp);
           pIdx = RandomSqc(pp);
%            pIdx = 314;
           cur_Class = Classinfo(pp);
           result(cur_Class,1) = result(cur_Class,1)+1;                     
           cur_Groups = zeros(1,nOutputs);
           cur_Groups((cur_Class-1)*nGroups+1:cur_Class*nGroups) = 1;
           cur_Tmax = TTmax(pp);
           T_size = ceil((cur_Tmax+2*tau_m)/dt);
           gT_size = gpuArray(int32(T_size));
           
           nstart = 1;
           for ilayer = 1:nLayers
               %-------------for input,output,u
                  if ilayer == 1
                      output{ilayer} = zeros(T_size,Structure.Layers(ilayer));
                      for i = 1:length(ptn{pIdx})
                            output{ilayer}(ptn{pIdx}(i,2),ptn{pIdx}(i,1)) = 1;
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
%            imagesc(output{2});
      end
      
end
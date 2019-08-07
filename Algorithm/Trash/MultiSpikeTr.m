function Structure = MultiSpikeTr(Data,ValData,maxEpoch,Output_neurons,nGroups,Layers,startEpoch,existweights,curStructure,path,tau_m)
    tic
    %Data:TrainData
    %ValData:TestData
    %maxEpoch:100
    %nGroups:10
    %Layers:~nAfferents*2
    %startEpoch:1
    %existweights = 0 or 1
    %curStructure = 0 or Structure
    %path = 0 or '/home/ncrc/projrcts/gpj/Spatio-temporal neuron model/MultiLayerSNN/Name.mat'
    %tau_m = [30e-3,30e-3]
    % Output_neurons :  size: (nOutput_types,nCls)  #desired spikes
    % Data.Tamx = get_Tmax2(Data.ptn)
    
    
    %----------params----------------------------
%     [Data,ValData] = divideData(Data,[6,1]);
    [nPtns,nAfferents] = size(Data.ptn);
    nCls = length(Data.Labels_name);
    Tmax = Data.Tmax;
    Structure.Output_neurons = Output_neurons;
    
    dt = 3e-3;
    a1 = gpuArray(double(2));
    % gtau = gpuArray(double(tau));
%     tau_m = 10e-3;
    Structure.tau_m = tau_m;
    tau_m = Structure.tau_m(1);
    tau_s = tau_m/4;
    beta = tau_m/tau_s;
    V0(1) = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
    gdecay1(1) = gpuArray(double(exp(-(dt/tau_m))));
    gdecay2(1) = gpuArray(double(exp(-(dt/tau_s))));
    
    tau_m = Structure.tau_m(2);
    tau_s = tau_m/4;
    beta = tau_m/tau_s;
    V0(2) = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
    gdecay1(2) = gpuArray(double(exp(-(dt/tau_m))));
    gdecay2(2) = gpuArray(double(exp(-(dt/tau_s))));
    
    lmd = 1e-2/gather(V0(1));
%     lmd(2) = 1e-5/gather(V0(1));
%     max_lmd = 1e-2/gather(V0(1));
%     min_lmd = 1e-3/gather(V0(1)); 
%     lmd_interval = max_lmd - min_lmd;
%     decay_steps = 100;
%     warm = 1;
%     lmds = 1 - (1:decay_steps)/decay_steps;
    
    
    
    best_result.accuracy = 0;
    tmpWeights = 0;
    gdt = gpuArray(double(dt));
    threshold = 1;
    gthreshold = gpuArray(double(threshold));
    subthreshold = gpuArray(double(0.3*threshold));

    ptn = ptn2df(Data.ptn,dt);

    ClassLabels = Data.Labels;
    Labels_name = Data.Labels_name;
    
   Output_desired = gpuArray(int32(repmat(Structure.Output_neurons,nGroups,1)));
    nOutput_types = size(Structure.Output_neurons,1);
    nOutputs = nOutput_types *nGroups;
         for ilabel = 1:nCls
            nLabels(ilabel) = length(find(ClassLabels == ilabel));
        end
        nLabels = repmat(nLabels,nOutput_types,1);
    % Structure.Layers = gpuArray(int8(Structure.Layers));
    
    nThreadsperBlock = 256;

    writeAcclog = 1;
    if existweights == 0
        Structure.Labels_name = Labels_name;
        Structure.Layers = gpuArray(int32([nAfferents,Layers,nOutputs]));
        Structure.gatherLayers = [nAfferents,Layers,nOutputs];
        nLayers= length(Structure.gatherLayers);
        Structure.AllWeights = cell(1,nLayers); 
        for ilayer = 2:nLayers
            Structure.AllWeights{ilayer} = gpuArray(double(...
                1e-2*randn(Structure.gatherLayers(ilayer-1),gather(Structure.gatherLayers(ilayer)))));
        end
           if writeAcclog == 1
            Acclog = zeros(maxEpoch,nOutput_types,nCls);
            resultlog = [];
           end
    else
        Structure = curStructure;
        nLayers= length(Structure.gatherLayers);
           if writeAcclog == 1
            Acclog = Structure.Acclog;
            resultlog = [];
           end
    end
    nClsMax = size(Data.actNumList,2);
    %------------------inital----------------------------------------------------
%     nLayers = length(Structure.Layers);
    DW = cell(1,nLayers); 
    input = cell(1,nLayers);
    output = cell(1,nLayers);
    %b = cell(1,nLayers);
    u = cell(1,nLayers);
    sumLayers = sum(Structure.Layers);
%     Structure.Acclog = cell(nCls+2,3);

    

    compute_u_kernel = parallel.gpu.CUDAKernel('compute_u_kernel.ptx','compute_u_kernel.cu');
    compute_u_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    compute_u_kernel_without_threshold = parallel.gpu.CUDAKernel('compute_u_kernel_without_threshold.ptx','compute_u_kernel_without_threshold.cu');
    compute_u_kernel_without_threshold.ThreadBlockSize = [nThreadsperBlock,1,1];
    t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev2.ptx','t_v_alter_multispikev2.cu');
    t_v_alter.ThreadBlockSize = [nThreadsperBlock,1,1];
    compute_outputDW_kernel = parallel.gpu.CUDAKernel('compute_outputDW_kernel.ptx','compute_outputDW_kernel.cu');
    compute_outputDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    compute_hiddenDW_kernel = parallel.gpu.CUDAKernel('compute_hiddenDW_kernel.ptx','compute_hiddenDW_kernel.cu');
    compute_hiddenDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    

    num_pos_sample = zeros(1,nCls);
    for icls = 1:nCls
        num_pos_sample(icls) = length(find(ClassLabels == icls));
    end
    
    

   isample = 1;
   for iepoch = startEpoch:maxEpoch
      RandomSqc=randperm(nPtns);  % random perm the training sequence.
      temp = zeros(1,nOutputs);
%       RandomSqc=1:nPtns;
%       Classinfo = ClassLabels(RandomSqc);
%       TTmax = Tmax(RandomSqc);
%       result = zeros(nCls+2,6);
%       err_Num = zeros(2,nOutputs); %pos,neg
      Hit = zeros(nOutputs,nCls);
      for  pp = 1:nPtns
           pIdx = RandomSqc(pp);
%            disp(pIdx);
%            pIdx = 314;
%            cur_Class = ClassLabels(pIdx);
%            result(cur_Class,1) = result(cur_Class,1)+1;
           tmp = find(Data.actNumList(:,pIdx));
           cur_clsidx = Data.actNumList(tmp,pIdx);
           desired= gpuArray(int32(sum(Output_desired(:,cur_clsidx),2)));
          
           
           
           cur_Tmax = Tmax(pIdx);
           T_size = ceil((cur_Tmax+2*tau_m)/dt);
           gT_size = gpuArray(int32(T_size));
           
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
                  input{ilayer} = output{ilayer-1}*Structure.AllWeights{ilayer}*V0(ilayer - 1);
                  output{ilayer} = gpuArray(zeros(T_size,Structure.Layers(ilayer),'double'));
                  u{ilayer} = gpuArray(zeros(T_size,Structure.Layers(ilayer),'double'));
                  if ilayer == nLayers
                      t_alter = gpuArray(zeros(nOutputs,1,'int32'));
                      direction = gpuArray(zeros(nOutputs,1,'double'));
                      t_v_alter.GridSize = [ceil(Structure.gatherLayers(ilayer)/nThreadsperBlock),1,1];
                      [output{ilayer},u{ilayer},t_alter,direction] = feval(t_v_alter,output{ilayer},u{ilayer},t_alter, direction,desired,input{ilayer},gT_size,...
                           Structure.Layers(ilayer),gdecay1(2),gdecay2(2),gthreshold,subthreshold);
                  else
                      compute_u_kernel.GridSize = [ceil(Structure.gatherLayers(ilayer)/nThreadsperBlock),1,1];
                      [output{ilayer},u{ilayer}] = feval(compute_u_kernel,u{ilayer},output{ilayer},input{ilayer},gT_size,...
                      gpuArray(int32(Structure.Layers(ilayer))),gdecay1(1),gdecay2(1),gthreshold);
                  end                 
%                    output{ilayer} = double(output{ilayer});
           end
           numActual =  sum(output{nLayers},1);
           temp = temp +abs(numActual - double(desired'));
           %change the weights of outputlayer
           [t_alter,t_idx] = sort(t_alter);
           tmp = find(t_alter);
           nT = length(tmp);
           if nT ~= 0
               nT = gpuArray(int32(nT));
               t_alter = t_alter(tmp);
               t_idx = int32(t_idx(tmp));
               direction = direction(t_idx);
               weight_toutput = Structure.AllWeights{nLayers}(:,t_idx);
               
               compute_outputDW_kernel.GridSize = [ceil(Structure.gatherLayers(nLayers - 1)/nThreadsperBlock),1,1];
               DW{nLayers} = gpuArray(zeros(Structure.Layers(nLayers-1),Structure.Layers(nLayers),'double'));
               DW{nLayers} = feval(compute_outputDW_kernel,DW{nLayers},output{nLayers - 1},Structure.Layers(nLayers -1)...
                   ,gT_size,direction,nT,t_alter,t_idx,gdecay1(2),gdecay2(2),V0(2));
               
               compute_hiddenDW_kernel.GridSize = [ceil(Structure.gatherLayers(nLayers - 2)/nThreadsperBlock),Structure.gatherLayers(nLayers-1),1];
               DW{nLayers-1} = gpuArray(zeros(Structure.Layers(nLayers-2),Structure.Layers(nLayers-1),'double'));
               DW{nLayers-1} = feval(compute_hiddenDW_kernel,DW{nLayers-1},u{nLayers-1},output{nLayers-2},Structure.Layers(nLayers-2),...
                   Structure.Layers(nLayers-1),gT_size,nT,t_alter,weight_toutput,direction,gdecay1,gdecay2,gthreshold,V0(1),a1);

               
%                isample = mod(isample,decay_steps);
%                if isample == 0
%                    isample = decay_steps;
%                    max_lmd = max_lmd*warm;
%                    lmd_interval = max_lmd -min_lmd;
%                end
%                lmd = lmd_interval*lmds(isample)+min_lmd;
%                isample = isample + 1;
               
               Structure.AllWeights{nLayers} = Structure.AllWeights{nLayers} + lmd*DW{nLayers};
               Structure.AllWeights{nLayers-1} = Structure.AllWeights{nLayers-1} + lmd*DW{nLayers-1};
%                tmp = t_idx(find(direction == 1));
%                err_Num(1,tmp)  = err_Num(1,tmp) + 1;
%                tmp = t_idx(find(direction == -1));
%                err_Num(2,tmp)  = err_Num(2,tmp) + 1;
           end
           
           tmp = find(direction == 0);
%            Hit(tmp,cur_Class) = Hit(tmp,cur_Class) + 1;
          if pp == nPtns
               fprintf('\n');
           end
           if mod(pp,ceil(nPtns/100))== 0
               fprintf('.');
           end
      end
      acc = temp/nPtns;
      acc = reshape(acc,nOutput_types,nGroups);
      acc = sum(acc,2);
      acc = acc/nGroups;
      acc = gather(acc);
       fprintf('\n');
      disp(['Epoch:',num2str(iepoch)]);
%       disp(['accuracy:']);
      acc = acc'
%       err_Num = squeeze(sum(reshape(err_Num,2,nGroups,nCls),2));
%       err_Num = err_Num/nGroups;
     
%       Hit = sum(reshape(Hit,nOutput_types,nGroups,nCls),2);
%       Hit = reshape(Hit,nOutput_types,nCls);
%       Hit = Hit/nGroups;
%       Acc = (Hit ./ nLabels);
% %       disp(iepoch);
%       Acclog(iepoch,:,:) = Acc;
%       disp(Data.Labels_name);
%       disp(Acc);
%       Structure.Acclog(iepoch,:,:)=Acc;
%       totalAcc = sum(Hit,2) ./ sum(nLabels,2);
%       disp(['total accuracy:',num2str(totalAcc)]);
%       if(1- totalAcc <= 1e-3)
%           disp('success!')
%           break;
%       end
      
%       pos_err = err_Num(1,:) ./num_pos_sample;
%       neg_err = err_Num(2,:) ./(nPtns - num_pos_sample);
%       total_err = (err_Num(1,:) + err_Num(2,:))/nPtns;
%       for icls = 1:nCls
% %           pos_err =  err_Num(1,icls)/num_pos_sample(icls);
% %           neg_err = err_Num(2,icls)/(nPtns - num_pos_sample(icls));
% %           total_err = (err_Num(1,icls) + err_Num(2,icls))/nPtns;
%          disp(['For ',Labels_name{icls},...
%            '   neurons: positive err :',num2str(pos_err(icls)),',  negative err :'...
%            num2str(neg_err(icls)),',  total err :',num2str(total_err(icls))]);
%       end
%       
%       if writeAcclog == 1
%           Acclog{1} = [Acclog{1};pos_err];
%           Acclog{2} = [Acclog{2};neg_err];
%           Acclog{3} = [Acclog{3};total_err];
%       end
% %       lmd = lmd*0.9;
%       Result = MultiSpikeTe(ValData,Structure,nGroups,0.7);
%       
%       
%       resultlog = [resultlog,Result.accuracy];
% %       Acclog{4} = [Acclog{4},Result.accuracy];
%       if Result.accuracy > best_result.accuracy
%           best_result = Result;
%           tmpWeights = Structure.AllWeights;
%       end
      
%       if (sum(total_err)/nCls) < 1e-4
%           disp('Success!')
%           break;
%       end
   end
   Structure.Labels_name = Labels_name;
   Structure.best_result = best_result;
   Structure.Acclog = Acclog;
%    Structure.AllWeights = tmpWeights;
   
   if path~= 0
       save(path,'Structure')
   end
   
%    save(path,'Structure');
   toc
end
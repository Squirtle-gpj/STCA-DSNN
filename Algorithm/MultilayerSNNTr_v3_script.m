%function Structure = MultilayerSNNTr_v3(Data,ValData,ValData2,Output_neurons,nGroups,maxEpoch,Layers,startEpoch,existweights,curStructure,path,tau_m,dropout)
    tic
%     run MultilayerSNN_params
    %Structure = MultilayerSNNTr_v3(Data,Data,get_output_neurons(10,1),1,100,[1200],1,0,0,0,30e-3,1)
    % Structure = MultilayerSNNTr_v3(trainData,trainData,o,1,30,[],1,0,0,0,[60e-3,60e-3],1)
    % Output_neurons :  size: (nOutput_types,nCls)  #desired spikes
    %----------params----------------------------
%     [Data,ValData] = divideData(Data,[6,1]);
    desired = gpuArray(int32(10));
    [nPtns,nAfferents] = size(Data.ptn);
    nCls = length(Data.Labels_name);
    Tmax = Data.Tmax;
    Structure.Output_neurons = Output_neurons;
    Structure.dropout = dropout;
    
%     dt = 3e-3;
    c = gpuArray(int32(1));
    a1 = gpuArray(double(2.01));
    % gtau = gpuArray(double(tau));
%     tau_m = 10e-3;
    Structure.tau_m = tau_m;
    tau_m = Structure.tau_m;
    maxsteps = getMaxsteps(tau_m,dt,1e-4);
    interval = gpuArray(int32(0.5*maxsteps));
    tau_s = tau_m/4;
    beta = tau_m/tau_s;
    V0 = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
    gdecay1 = gpuArray(double(exp(-(dt/tau_m))));
    gdecay2 = gpuArray(double(exp(-(dt/tau_s))));
    
%     tau_m = Structure.tau_m(2);
%     tau_s = tau_m/4;
%     beta = tau_m/tau_s;
%     V0(2) = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
%     gdecay1(2) = gpuArray(double(exp(-(dt/tau_m))));
%     gdecay2(2) = gpuArray(double(exp(-(dt/tau_s))));
    
    
%     lmd(2) = 1e-5/gather(V0(1));
%     max_lmd = 1e-2/gather(V0(1));
%     min_lmd = 1e-3/gather(V0(1)); 
%     lmd_interval = max_lmd - min_lmd;
%     decay_steps = 100;
%     warm = 1;
%     lmds = 1 - (1:decay_steps)/decay_steps;
    
    
    
    best_result = 9999;
    best_result2.accuracy = 0;
    tmpWeights = 0;
    gdt = gpuArray(double(dt));
    threshold = 1;
    gthreshold = gpuArray(double(threshold));
    subthreshold = gpuArray(double(0.3*threshold));

    ptn = ptn2df(Data.ptn,dt);

%     ClassLabels = Data.Labels;
    Labels_name = Data.Labels_name;
    
    Output_desired = gpuArray(int32(repmat(Structure.Output_neurons,nGroups,1)));
    nOutput_types = size(Structure.Output_neurons,1);
    nOutputs = nOutput_types *nGroups;
%          for ilabel = 1:nCls
%             nLabels(ilabel) = length(find(ClassLabels == ilabel));
%         end
%         nLabels = repmat(nLabels,nOutput_types,1);
    
    % Structure.Layers = gpuArray(int8(Structure.Layers));
    
    nThreadsperBlock = 256;
    nThreadsperBlock_sqrt = 32;

    writeErrlog = 1;
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
           if writeErrlog == 1
            Errlog = cell(3,1);
           end
    else
        Structure = curStructure;
        nLayers= length(Structure.gatherLayers);
           if writeErrlog == 1
            Errlog = Structure.Errlog;
           end
    end
    lmd = zeros(nLayers,1);
    lmd(nLayers) = 1e-2/gather(V0);
    for ilayer = nLayers-1:-1:1
        lmd(ilayer) = lmd(ilayer+1)/gather(V0);
    end
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
%     t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_classification.ptx','t_v_alter_classification.cu');
%     t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev5.ptx','t_v_alter_multispikev5.cu');
    t_v_alter.ThreadBlockSize = [nThreadsperBlock,1,1];
%     compute_outputDW_kernel = parallel.gpu.CUDAKernel('compute_outputDW_kernel.ptx','compute_outputDW_kernel.cu');
%     compute_outputDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
%     compute_hiddenDW_kernel = parallel.gpu.CUDAKernel('compute_hiddenDW_kernel.ptx','compute_hiddenDW_kernel.cu');
%     compute_hiddenDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    
    geth = parallel.gpu.CUDAKernel('geth.ptx','geth.cu');
    geth.ThreadBlockSize = [nThreadsperBlock_sqrt,nThreadsperBlock_sqrt,1];
%     dVdM_lastlayer = parallel.gpu.CUDAKernel('dVdM_lastlayer.ptx','dVdM_lastlayer.cu');
%     dVdM_lastlayer.ThreadBlockSize = [nThreadsperBlock,1,1];
    dVdO_lastlayer = parallel.gpu.CUDAKernel('dVdO_lastlayer.ptx','dVdO_lastlayer.cu');
    dVdO_lastlayer.ThreadBlockSize = [nThreadsperBlock_sqrt,nThreadsperBlock_sqrt,1];
%     dVdM_hiddenlayer = parallel.gpu.CUDAKernel('dVdM_hiddenlayer.ptx','dVdM_hiddenlayer.cu');
%     dVdM_hiddenlayer.ThreadBlockSize = [nThreadsperBlock,1,1];
    dVdinput_lastlayer = parallel.gpu.CUDAKernel('dVdinput_lastlayer_m3.ptx','dVdinput_lastlayer_m3.cu');
    dVdinput_lastlayer.ThreadBlockSize = [nThreadsperBlock,1,1];
    dVdinput_hiddenlayer = parallel.gpu.CUDAKernel('dVdinput_hiddenlayer_m3.ptx','dVdinput_hiddenlayer_m3.cu');
    dVdinput_hiddenlayer.ThreadBlockSize = [nThreadsperBlock_sqrt,nThreadsperBlock_sqrt,1];

    dVdO_hiddenlayer = parallel.gpu.CUDAKernel('dVdO_hiddenlayer.ptx','dVdO_hiddenlayer.cu');
    dVdO_hiddenlayer.ThreadBlockSize = [nThreadsperBlock_sqrt,nThreadsperBlock_sqrt,1];
    
%     num_pos_sample = zeros(1,nCls);
%     for icls = 1:nCls
%         num_pos_sample(icls) = length(find(ClassLabels == icls));
%     end
    
 
%    nDrop = gpuArray(zeros(Structure.gatherLayers*dropout,'int32'));
   nDrop = ceil(Structure.gatherLayers*dropout);
   nDrop(1) = nAfferents;
   
   isample = 1;
   for iepoch = startEpoch:maxEpoch
      RandomSqc=randperm(nPtns);  % random perm the training sequence.
      temp = zeros(1,nOutputs);
%       RandomSqc=1:nPtns;
%       Classinfo = ClassLabels(RandomSqc);
%       TTmax = Tmax(RandomSqc);
      result = zeros(nCls+2,6);
      err_Num = zeros(2,nOutputs); %pos,neg
      for  pp = 1:nPtns
          nDrop(nLayers) = nOutputs;
          pIdx = RandomSqc(pp);
             pIdx = pp;
%            disp(pIdx);
%            pIdx = 314;
           tmp = find(Data.actNumList(:,pIdx));
           cur_clsidx = Data.actNumList(tmp,pIdx);
           desired= gpuArray(int32(sum(Output_desired(:,cur_clsidx),2)));
           cur_Tmax = Tmax(pIdx);
           T_size = ceil((cur_Tmax+2*tau_m)/dt);
           gT_size = gpuArray(int32(T_size));
           
           %% forward
           Drop = cell(nLayers,1);
           Drop{1} = 1:nAfferents;
           Drop{nLayers} = 1:nOutputs;
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
                  if ilayer~= nLayers
                    Drop{ilayer} = randperm(Structure.gatherLayers(ilayer),nDrop(ilayer));
%                     Drop{ilayer} = 1:nDrop(ilayer);
                  end
                  input{ilayer} = output{ilayer-1}*Structure.AllWeights{ilayer}(Drop{ilayer -1},Drop{ilayer})*V0;
                  output{ilayer} = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
                  u{ilayer} = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
                  if ilayer == nLayers
%                       t_alter = gpuArray(zeros(nOutputs,1,'int32'));
%                       direction = gpuArray(zeros(nOutputs,1,'double'));
%                       t_v_alter.GridSize = [ceil(nOutputs/nThreadsperBlock),1,1];
%                       int_param = [gT_size,nDrop(ilayer)];
%                       param = [gthreshold,subthreshold,gdecay1(1),gdecay2(1)];
%                       [output{ilayer},u{ilayer},t_alter,direction] = feval(t_v_alter,output{ilayer},u{ilayer},t_alter, direction,desired,input{ilayer},int_param,param);
                          t_alter = gpuArray(zeros(nOutputs,1,'int32'));
                      direction = gpuArray(zeros(nOutputs,1,'double'));
                      t_v_alter.GridSize = [ceil(Structure.gatherLayers(ilayer)/nThreadsperBlock),1,1];
                      numActual = gpuArray(int32(zeros(1,nOutputs)));
                      [output{ilayer},u{ilayer},numActual,t_alter,direction] = feval(t_v_alter,output{ilayer},u{ilayer},numActual,t_alter, direction,desired,input{ilayer},gT_size,...
                           Structure.Layers(ilayer),gdecay1(1),gdecay2(1),gthreshold,interval);
                  else
                      int_param = [gT_size,nDrop(ilayer)];
                      param = [gthreshold,gdecay1(1),gdecay2(1)];
                      compute_u_kernel.GridSize = [ceil(nDrop(ilayer)/nThreadsperBlock),1,1];
                      [output{ilayer},u{ilayer}] = feval(compute_u_kernel,u{ilayer},output{ilayer},input{ilayer},int_param,param);
                  end                 
%                    output{ilayer} = double(output{ilayer});
           end
%            numActual =  sum(output{nLayers},1);

           temp = temp +abs(double(numActual) - double(desired'));
           %% backpropagation
           
           
          tmp = find(direction~=0);
           nT = length(tmp);
           if nT ~= 0
%                nT = gpuArray(int32(nT));
               t_alter = t_alter(tmp);
               for it = 1:length(t_alter)
                   if(t_alter(it) == T_size)
                       t_alter(it) =ceil(rand(1,1)*T_size); 
                   end
               end
               t_alter_max = gather(max(t_alter));
%                [t_alter,t_idx] = sort(t_alter);
%                t_idx = int32(t_idx);
               direction = direction(tmp);
               Drop{nLayers} = tmp;
               nDrop(nLayers) = nT;
               u{nLayers} = u{nLayers}(:,tmp);
               for ilayer = nLayers:-1:2
                   
                   geth.GridSize = [ceil(t_alter_max/nThreadsperBlock_sqrt),ceil(nDrop(ilayer)/nThreadsperBlock_sqrt),1];
                   h = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
                   int_param = [gT_size,nDrop(ilayer),c];
                   param = [a1,gthreshold];
                   h = feval(geth,h,u{ilayer},int_param,param);
                   
                   if ilayer == nLayers
%                        dVdM = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
                       dVdinput = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
                       dVdinput_lastlayer.GridSize =  [ceil(nDrop(ilayer)/nThreadsperBlock),1,1];
                       dVdinput = feval(dVdinput_lastlayer,dVdinput,h,t_alter,direction,[gthreshold,gdecay1,gdecay2],[gT_size,nDrop(ilayer),maxsteps]);
                       
%                        dVdM = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
%                        dVdS = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
%                        dVdM_lastlayer.GridSize =  [ceil(nDrop(ilayer)/nThreadsperBlock),1,1];
%                        [dVdM,dVdS] = feval(dVdM_lastlayer,dVdM,dVdS,h,t_alter,direction,[gthreshold,gdecay1,gdecay2],[gT_size,nDrop(ilayer)]);
%                        
% %                        clear h dVdinput
%                        dVdinput = dVdM - dVdS;
                       clear h 
%                        dVdinput = dVdM - dVdS;
%                        clear dVdM dVdS
%                        tmp = sum(dVdinput(:,find(direction == 1),:),2) - sum(dVdinput(:,find(direction == -1),:),2);
                       DW{ilayer} = (output{ilayer - 1}' * dVdinput).*V0;
%                        Structure.AllWeights{ilayer}(Drop{ilayer-1},Drop{ilayer}) = Structure.AllWeights{ilayer }(Drop{ilayer-1},Drop{ilayer}) + lmd*tmp;
                       clear tmp 
%                        disp(1)
                   else
                       dVdO = gpuArray(zeros(T_size,nDrop(ilayer),nDrop(nLayers),'double'));
                       if ilayer == (nLayers-1)
                           dVdO_lastlayer.GridSize = [ceil(nDrop(ilayer)/nThreadsperBlock_sqrt),ceil(nDrop(nLayers)/nThreadsperBlock_sqrt),max(maxsteps,t_alter_max)];
                           dVdO = feval(dVdO_lastlayer,dVdO,dVdinput,Structure.AllWeights{ilayer+1}(Drop{ilayer},Drop{ilayer+1}),t_alter,V0,[gT_size,nDrop(ilayer),nDrop(nLayers),maxsteps]);
                       else
                           dVdO_hiddenlayer.GridSize = [ceil(nDrop(ilayer)/nThreadsperBlock_sqrt),ceil(nDrop(nLayers)/nThreadsperBlock_sqrt),max(maxsteps,t_alter_max)];
                           dVdO = feval(dVdO_hiddenlayer,dVdO,dVdinput,Structure.AllWeights{ilayer+1}(Drop{ilayer},Drop{ilayer+1}),t_alter,V0,[gT_size,nDrop(ilayer),nDrop(ilayer+1),nDrop(nLayers),maxsteps]);
                       end
                       dVdinput_hiddenlayer.GridSize = [ceil(nDrop(ilayer)/nThreadsperBlock_sqrt),ceil(nDrop(nLayers)/nThreadsperBlock_sqrt)];
                       dVdinput = gpuArray(zeros(T_size,nDrop(ilayer),nDrop(nLayers),'double'));
%                        dVdS = gpuArray(zeros(T_size,nDrop(ilayer),nDrop(nLayers),'double'));
                       dVdinput = feval(dVdinput_hiddenlayer,dVdinput,dVdO,h,t_alter,[gthreshold,gdecay1,gdecay2],[gT_size,nDrop(ilayer),nDrop(nLayers),maxsteps]);
                       
                       clear h dVdO 
%                        dVdinput = dVdM - dVdS;
%                        clear dVdM dVdS
                       tmp = sum(dVdinput,3);
                       DW{ilayer} = (output{ilayer - 1}' * squeeze(tmp)).*V0;
                       
%                        Structure.AllWeights{ilayer }(Drop{ilayer-1},Drop{ilayer}) = Structure.AllWeights{ilayer }(Drop{ilayer-1},Drop{ilayer}) + lmd*tmp;
                       clear tmp 
%                        disp(2)
                   end
                   
               end
               for ilayer = nLayers:-1:2
                   Structure.AllWeights{ilayer}(Drop{ilayer-1},Drop{ilayer}) = Structure.AllWeights{ilayer }(Drop{ilayer-1},Drop{ilayer}) + lmd(ilayer)*DW{ilayer};
               end
%                weight_toutput = Structure.AllWeights{nLayers}(:,t_idx);
%                
%                compute_outputDW_kernel.GridSize = [ceil(Structure.gatherLayers(nLayers - 1)/nThreadsperBlock),1,1];
%                DW{nLayers} = gpuArray(zeros(Structure.Layers(nLayers-1),Structure.Layers(nLayers),'double'));
%                DW{nLayers} = feval(compute_outputDW_kernel,DW{nLayers},output{nLayers - 1},Structure.Layers(nLayers -1)...
%                    ,gT_size,direction,nT,t_alter,t_idx,gdecay1(2),gdecay2(2),V0(2));
%                
%                compute_hiddenDW_kernel.GridSize = [ceil(Structure.gatherLayers(nLayers - 2)/nThreadsperBlock),Structure.gatherLayers(nLayers-1),1];
%                DW{nLayers-1} = gpuArray(zeros(Structure.Layers(nLayers-2),Structure.Layers(nLayers-1),'double'));
%                DW{nLayers-1} = feval(compute_hiddenDW_kernel,DW{nLayers-1},u{nLayers-1},output{nLayers-2},Structure.Layers(nLayers-2),...
%                    Structure.Layers(nLayers-1),gT_size,nT,t_alter,weight_toutput,direction,gdecay1,gdecay2,gthreshold,V0(1),a1);
% 
%                
% %                isample = mod(isample,decay_steps);
% %                if isample == 0
% %                    isample = decay_steps;
% %                    max_lmd = max_lmd*warm;
% %                    lmd_interval = max_lmd -min_lmd;
% %                end
% %                lmd = lmd_interval*lmds(isample)+min_lmd;
% %                isample = isample + 1;
%                
%                Structure.AllWeights{nLayers} = Structure.AllWeights{nLayers} + lmd*DW{nLayers};
%                Structure.AllWeights{nLayers-1} = Structure.AllWeights{nLayers-1} + lmd*DW{nLayers-1};
               tmp = Drop{nLayers}(find(direction == 1));
               err_Num(1,tmp)  = err_Num(1,tmp) + 1;
               tmp = Drop{nLayers}(find(direction == -1));
               err_Num(2,tmp)  = err_Num(2,tmp) + 1;
           end
          if pp == nPtns
               fprintf('\n');
           end
           if mod(pp,ceil(nPtns/100))== 0
               fprintf('.');
           end
%            if mod(pp,ceil(nPtns/1000))== 0
%                fprintf('\n');
%                Result2 = MultilayerSNNTe_v2(ValData2,Structure,nGroups);
%                Errlog{3} = [Errlog{3},Result2.accuracy]
%            end
           
%            fprintf(num2str(pp));
%            gather(output{nLayers})

%         %% experiment1 observing the output spikes with the number of input sample increasing
%           
%            cur_Tmax = spliced_Tmax;
%            T_size = ceil((cur_Tmax+2*tau_m)/dt);
%            gT_size = gpuArray(int32(T_size));
%            %% forward
%            for ilayer = 1:nLayers
%                %-------------for input,output,u
%                   if ilayer == 1
%                       output{ilayer} = zeros(T_size,Structure.Layers(ilayer));
%                       for i = 1:length(spliced_ptn{1})
%                             output{ilayer}(spliced_ptn{1}(i,2),spliced_ptn{1}(i,1)) = 1;
%                       end
%                        output{ilayer} = gpuArray(double(output{ilayer}));%T_size * Ncurlayer
%                        continue;
%                   end
%                   if ilayer~= nLayers
%                     Drop{ilayer} = randperm(Structure.gatherLayers(ilayer),nDrop(ilayer));
%                   end
%                   input{ilayer} = output{ilayer-1}*Structure.AllWeights{ilayer}(Drop{ilayer -1},Drop{ilayer})*V0;
%                   output{ilayer} = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
%                   u{ilayer} = gpuArray(zeros(T_size,nDrop(ilayer),'double'));
% 
%                       int_param = [gT_size,nDrop(ilayer)];
%                       param = [gthreshold,gdecay1(1),gdecay2(1)];
%                       compute_u_kernel.GridSize = [ceil(nDrop(ilayer)/nThreadsperBlock),1,1];
%                       [output{ilayer},u{ilayer}] = feval(compute_u_kernel,u{ilayer},output{ilayer},input{ilayer},int_param,param);
%            end
%            [Tspike,ineuron] = find(output{nLayers});
%            Ninput_samples = Ninput_samples + 1;
%            output_spike_timing = [output_spike_timing;[Tspike,Ninput_samples*ones(length(Tspike),1),ineuron]];
      end
%       err_Num = squeeze(sum(reshape(err_Num,2,nGroups,nCls),2));
%       err_Num = err_Num/nGroups;
%       fprintf('\n');
%       disp(['Epoch:',num2str(iepoch)]);
%       
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
      acc = temp/nPtns;
      acc = reshape(acc,nOutput_types,nGroups);
      acc = sum(acc,2);
      acc = acc/nGroups;
      acc = gather(acc);
       fprintf('\n');
      disp(['Epoch:',num2str(iepoch)]);
%       disp(['accuracy:']);
      acc = acc'
%       Result = MultilayerSNNTe_v3(ValData,Structure,nGroups);
      Result = 0;  
      Result2 = MultilayerSNNTe_v2(ValData2,Structure,nGroups);
      if writeErrlog == 1
          Errlog{1} = [Errlog{1};acc];
          Errlog{2} = [Errlog{2};Result];
          Errlog{3} = [Errlog{3},Result2.accuracy];
      end
%       lmd = lmd*0.9;
       
%       
%       Errlog{4} = [Errlog{4},Result.accuracy];
%       if (sum(Result)/nOutputs) < (sum(best_result)/nOutputs)
%           best_result = Result;
%           tmpWeights = Structure.AllWeights;
%       end


      if Result2.accuracy > best_result2.accuracy
          best_result2 = Result2;
          Structure.tmpWeights = Structure.AllWeights;
      end
      
%       if (sum(total_err)/nCls) < 1e-4
%           disp('Success!')
%           break;
%       end
   end
   
   Structure.Labels_name = Labels_name;
   Structure.best_result = best_result;
   Structure.best_result2 = best_result2;
   Structure.Errlog = Errlog;
   if path~= 0
       save(path,'Structure')
   end
   
%    save(path,'Structure');
   toc
%end
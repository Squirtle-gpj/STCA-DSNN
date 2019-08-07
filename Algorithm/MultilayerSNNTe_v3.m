function Result = MultilayerSNNTe_v3(Data,Structure,nGroups)
    tic
    %Structure = MultilayerSNNTr_v3(Data,Data,get_output_neurons(10,1),1,100,[1200],1,0,0,0,30e-3,1)
    % Structure = MultilayerSNNTr_v3(trainData,trainData,o,1,30,[],1,0,0,0,[60e-3,60e-3],1)
    % Output_neurons :  size: (nOutput_types,nCls)  #desired spikes
    %----------params----------------------------
%     [Data,ValData] = divideData(Data,[6,1]);
%     desired = gpuArray(int32(10));
    [nPtns,nAfferents] = size(Data.ptn);
    nCls = length(Data.Labels_name);
    Tmax = Data.Tmax;
%     Structure.Output_neurons = Output_neurons;
%     Structure.dropout = dropout;
    
    dt = 3e-3;
    c = gpuArray(int32(1));
    a1 = gpuArray(double(2.01));
    % gtau = gpuArray(double(tau));
%     tau_m = 10e-3;
%     Structure.tau_m = tau_m;
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
    
    
    
    best_result.accuracy = 0;
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


        nLayers= length(Structure.gatherLayers);
%            if writeErrlog == 1
%             Errlog = Structure.Errlog;
%            end

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
    t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev5.ptx','t_v_alter_multispikev5.cu');
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
   dropout = 1;
   nDrop = ceil(Structure.gatherLayers*dropout);
   nDrop(1) = nAfferents;
   
   isample = 1;

%       RandomSqc=randperm(nPtns);  % random perm the training sequence.
      temp = zeros(1,nOutputs);
      RandomSqc=1:nPtns;
%       Classinfo = ClassLabels(RandomSqc);
%       TTmax = Tmax(RandomSqc);
      result = zeros(nCls+2,6);
      err_Num = zeros(2,nOutputs); %pos,neg
      for  pp = 1:nPtns
          nDrop(nLayers) = nOutputs;
           pIdx = RandomSqc(pp);
%                pIdx = pp;
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
  
               
           if pp == nPtns
               fprintf('\n');
           end
           if mod(pp,ceil(nPtns/100))== 0
               fprintf('.');
           end
      end
           

%            fprintf(num2str(pp));

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
%        fprintf('\n');
%       disp(['Epoch:',num2str(iepoch)]);
%       disp(['accuracy:']);
      Result = acc'
      Result = acc';
%       Result = MultilayerSNNTe_v2(ValData,Structure,nGroups);
%       if writeErrlog == 1
%           Errlog = [Errlog;acc];
%       end
%       lmd = lmd*0.9;
%       Result = MultilayerSNNTe_v2(ValData,Structure,nGroups);
%       
%       Errlog{4} = [Errlog{4},Result.accuracy];
%       if Result.accuracy > best_result.accuracy
%           best_result = Result;
%           tmpWeights = Structure.AllWeights;
%       end
      
%       if (sum(total_err)/nCls) < 1e-4
%           disp('Success!')
%           break;
%       end
   end
%    Structure.Labels_name = Labels_name;
% %    Structure.best_result = best_result;
%    Structure.Errlog = Errlog;
%    Structure.AllWeights = tmpWeights;
%    if path~= 0
%        save(path,'Structure')
%    end
   
%    save(path,'Structure');
%    toc
% end
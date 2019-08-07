function Result = MultilayerTempotronTe_v2(Data,Structure,testGroup)
    tic
    
    %----------params----------------------------
     nOutputs = Structure.gatherLayers(end);
    nCls = length(Data.Labels_name);
    nGroups = nOutputs/nCls;
    [nPtns,nAfferents] = size(Data.ptn);
%     nCls = length(Data.Labels_name);
    Tmax = Data.Tmax;
    
    dt = 3e-3;
%     a1 = gpuArray(double(2));
    % gtau = gpuArray(double(tau));
    
    tau_m = Structure.tau_m;
%     tau_s = tau_m/4;
    beta = 4;
    V0(1) = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
%     gdecay1(1) = gpuArray(double(exp(-(dt/tau_m))));
%     gdecay2(1) = gpuArray(double(exp(-(dt/tau_s))));
%     
%     gdecay1 = gpuArray(double(zeros(nLayers,1)));
%     gdecay2 = gpuArray(double(zeros(nLayers,1)));
%     lmd(nLayers) = 1e-2/gather(V0);
%     
%     for ilayer = nLayers-1:-1:1
%         lmd(ilayer) = lmd(ilayer+1)/gather(V0);
%         maxsteps(ilayer) = getMaxsteps(tau_m(ilayer),dt,1e-3);
%     %     maxsteps = 500;
%         tau_s = tau_m(ilayer)/4;
% 
%         gdecay1(ilayer) = gpuArray(double(exp(-(dt/tau_m(ilayer)))));
%         gdecay2(ilayer) = gpuArray(double(exp(-(dt/tau_s))));
%     end
    
%     tau_m = Structure.tau_m;
%     tau_s = tau_m/4;
%     beta = tau_m/tau_s;
%     V0(2) = gpuArray(double((1/(beta-1))*(beta^(beta/(beta-1)))));
%     gdecay1(2) = gpuArray(double(exp(-(dt/tau_m))));
%     gdecay2(2) = gpuArray(double(exp(-(dt/tau_s))));
    
%     lmd = 1e-2/gather(V0);
%     max_lmd = 1;
%     min_lmd = 1e-2/gather(V0); 
%     lmd_interval = max_lmd - min_lmd;
%     decay_steps = 100;
%     warm = 0.2;
%     lmds = 1 - (1:decay_steps)/decay_steps;
    
    
    
    
    gdt = gpuArray(double(dt));
    threshold = 1;
    gthreshold = gpuArray(double(threshold));
    ptn = ptn2df(Data.ptn,dt);

    ClassLabels = Data.Labels;
    Labels_name = Data.Labels_name;
%     nOutputs = nCls*nGroups;
%     Structure.Layers = gpuArray(int32([nAfferents,Layers,nOutputs]));
%     Structure.gatherLayers = [nAfferents,Layers,nOutputs];
%     % Structure.Layers = gpuArray(int8(Structure.Layers));
    nLayers = length(Structure.Layers);
    nThreadsperBlock = 256;
%     if ~isexist(Structure.dropout)
%         Structure.dropout = 1;
%     end
%     for ilayer = 2:nLayers
%         Structure.AllWeights{ilayer} = Structure.AllWeights{ilayer}*Structure.dropout;
%     end
    
    gdecay1 = gpuArray(double(zeros(nLayers,1)));
    gdecay2 = gpuArray(double(zeros(nLayers,1)));
    
    for ilayer = nLayers:-1:1
        
        tau_s = tau_m(ilayer)/4;

        gdecay1(ilayer) = gpuArray(double(exp(-(dt/tau_m(ilayer)))));
        gdecay2(ilayer) = gpuArray(double(exp(-(dt/tau_s))));
    end


    %------------------inital----------------------------------------------------
 
    input = cell(1,nLayers);
    output = cell(1,nLayers);
    %b = cell(1,nLayers);
    u = cell(1,nLayers);
    sumLayers = sum(Structure.Layers);
%     Structure.Acclog = cell(nCls+2,3);


    compute_u_kernel = parallel.gpu.CUDAKernel('compute_u_kernel_v1.ptx','compute_u_kernel_v1.cu');
    compute_u_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    compute_u_kernel_without_threshold = parallel.gpu.CUDAKernel('compute_u_kernel_without_threshold.ptx','compute_u_kernel_without_threshold.cu');
    compute_u_kernel_without_threshold.ThreadBlockSize = [nThreadsperBlock,1,1];
%     t_v_alter = parallel.gpu.CUDAKernel('t_v_alter.ptx','t_v_alter.cu');
%     t_v_alter.ThreadBlockSize = [nThreadsperBlock,1,1];
%     compute_outputDW_kernel = parallel.gpu.CUDAKernel('compute_outputDW_kernel.ptx','compute_outputDW_kernel.cu');
%     compute_outputDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
%     compute_hiddenDW_kernel = parallel.gpu.CUDAKernel('compute_hiddenDW_kernel.ptx','compute_hiddenDW_kernel.cu');
%     compute_hiddenDW_kernel.ThreadBlockSize = [nThreadsperBlock,1,1];
    




     confusion_matrix = zeros(nCls,nCls);
     Result.record = zeros(nCls+1,3);
     Result.Labels_name = Labels_name;
     sum_true = zeros(nCls,1);
      for  pp = 1:nPtns
           true_label = ClassLabels(pp); 
           sum_true(true_label) = sum_true(true_label)+1;
           cur_Tmax = Tmax(pp);
           T_size = ceil((cur_Tmax+2*tau_m(nLayers))/dt);
           gT_size = gpuArray(int32(T_size));
           
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
                  if ilayer == nLayers
                      compute_u_kernel_without_threshold.GridSize = [ceil(Structure.gatherLayers(ilayer)/nThreadsperBlock),1,1];
                      u{ilayer} = feval(compute_u_kernel_without_threshold,u{ilayer},input{ilayer},gT_size,...
                      gpuArray(int32(Structure.Layers(ilayer))),gdecay1(ilayer),gdecay2(ilayer));
                  else
                      compute_u_kernel.GridSize = [ceil(Structure.gatherLayers(ilayer)/nThreadsperBlock),1,1];
                      [output{ilayer},u{ilayer}] = feval(compute_u_kernel,u{ilayer},output{ilayer},input{ilayer},gT_size,...
                      gpuArray(int32(Structure.Layers(ilayer))),gdecay1(ilayer),gdecay2(ilayer),gthreshold);
                  end
    
                      
            
           end
           %strategy1
%            [~,predict_label] = max(sum(reshape(sum(output{nLayers}),nGroups,nCls)));
%            confusion_matrix(true_label,predict_label) = confusion_matrix(true_label,predict_label) + 1;
           %strategy2
           tmp = gather(max(u{nLayers}));
           tmp = reshape(tmp,nGroups,nCls);
           tmp = tmp(1:testGroup,:);
           tmp = sum(tmp,1);
           [~,predict_label] = max(tmp);
           confusion_matrix(true_label,predict_label) = confusion_matrix(true_label,predict_label) + 1;
           %strategy3 
%             tmp = reshape(sum(output{nLayers}),nGroups,nCls);
%             tmp(find(tmp)) = 1;
%             tmp = sum(tmp);
%             if sum(tmp) ~= 0
%                 [maxScore,predict_label] = max(tmp);
%                 idx = find(tmp == maxScore);
%                 if length(idx) ~= 1
%                     [~,predict_label] = max(sum(reshape(sum(output{nLayers}),nGroups,nCls)));
%                 end
%                 confusion_matrix(true_label,predict_label) = confusion_matrix(true_label,predict_label) + 1;
%             end
%                 
%            [~,predict_label] = max(sum(tmp));
           
           
           
           

          if pp == nPtns
               fprintf('\n');
           end
           if mod(pp,ceil(nPtns/100))== 0
               fprintf('.');
           end
      end
      fprintf('\n');
%       sum_true = sum(confusion_matrix,2);
      sum_predict = sum(confusion_matrix,1);
      tmp = 0;
      for icls = 1:nCls
          tmp = tmp + confusion_matrix(icls,icls);
          recall = confusion_matrix(icls,icls)/sum_true(icls);
          precision = confusion_matrix(icls,icls)/sum_predict(icls);
          F1 = 2*precision*recall/(precision + recall);
          disp([Labels_name{icls},':',' Recall: ',num2str(recall),' Precision: ',num2str(precision),' F1: ',num2str(F1),]);
          Result.record(icls,:) = [recall,precision,F1];
      end
      Result.record(nCls + 1,:) = sum(Result.record)/nCls;
      disp(['Total :',' Recall: ',num2str(Result.record(nCls + 1,1)),' Precision: ',num2str(Result.record(nCls + 1,2)),' F1: ',num2str(Result.record(nCls + 1,3))]);
      Result.accuracy = tmp/nPtns;
      disp(['Accuracy: ', num2str(Result.accuracy)]);
      Result.cfs_mtrx = confusion_matrix;
      imagesc(confusion_matrix);
   toc
end
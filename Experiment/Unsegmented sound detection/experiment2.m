% experiment2: the curvergence curve
% t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev5.ptx','t_v_alter_multispikev5.cu');
E2 = cell(6,1);
for i_test = 1:10
    maxEpoch = 100;
    Layers = [300];
    desired = 1;
    %v5 is the our loss function for solving TCA problem, v6 is the loss
    %function of MPD-AL (Malu Zhang, AAAI'19)
    t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev5.ptx','t_v_alter_multispikev5.cu');
    run MultilayerSNN_params
    clear Structure
    E2{1} = [E2{1};Errlog{3}];

    for i_spkcount = 2:6
            maxEpoch = 100;
            Layers = [];
            desired = 2*i_spkcount - 1;
            t_v_alter = parallel.gpu.CUDAKernel('t_v_alter_multispikev6.ptx','t_v_alter_multispikev6.cu');
            run MultilayerSNN_params
            clear Structure
            E2{i_spkcount} = [E2{i_spkcount};Errlog{3}];
    end
end


% for i = d 
% desired = i;
% run MultilayerSNN_params
% save(['aggregate-label/RWCP_10_IJCAI_128_TDP1_errlog_desired',7,'.mat'],'Errlog');
% end
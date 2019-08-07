% this function is usually applied to setting the parameter 'Output_neurons' in 'MultilayerSNNTr_v3.m'
function o = get_output_neurons(nCls,desired)
        o = zeros(nCls,nCls);
        for icls = 1:nCls
            o(icls,icls) = desired;
        end
end
% Converting the spike trains generated  by the 'generate_spikes' into
% specific spike timings. The format is same as the format in 'Data.ptn'.
% dt: the length of each time step.
function ptn = st2ptn(spike_trains,dt)
    ptn = cell(1,size(spike_trains,2));
    nAfferents = size(spike_trains,2);
    for iaff = 1:nAfferents
        tmp = find(spike_trains(:,iaff)~=0);
        if ~isempty(tmp)
            ptn{1,iaff} = tmp * dt;
        else
            ptn{1,iaff} = [];
        end
    end
end
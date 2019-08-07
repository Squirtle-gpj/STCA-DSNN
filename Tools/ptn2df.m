
% converting Data.ptn to other formats, the fist column records the indexes
% of fired neurons, the second columns records the ceil(spike time/ dt), dt
% is the length of each time step, unit: second
function df = ptn2df(ptn,dt)
    [nPtns,nAfferents] = size(ptn);
    df = cell(nPtns,1);
    for iptn = 1:nPtns
           firedTime = [];
           firedIndex = [];
           for iaff = 1:nAfferents
           
               firedtime = ptn{iptn,iaff};
               if ~isempty(firedtime) 
                if size(firedtime,1) == 1
                    firedTime = [firedTime;firedtime'];
                else
                    firedTime = [firedTime;firedtime];
                end
                firedIndex = [firedIndex;iaff*ones(length(firedtime),1)];
               end
           end
           firedTime = ceil(firedTime/dt);
%            firedIndex(firedTime == 0) = [];
           firedTime(firedTime == 0) = 1;
           
           df{iptn}=[firedIndex,firedTime];
    end

end
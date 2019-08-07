% splice length(s) patterns into 1 long patterns
% ptn: the total patterns, the format is same as the Data.ptn
% s: [idx1,idx2,idx3,...idxn], the indexes of the chosen independent patterns in the 'ptn'
% interval: the interval between two ajacent patterns, unit: s
% spliced_ptn: the spliced long patterns
% the  time point of last spike  in the 'spliced_ptn'
function [spliced_ptn, Tmax] = splice_ptn(ptn,s,interval)
    [nPtns,nAfferents] = size(ptn);
    spliced_ptn = cell(1,nAfferents);
    T_start = 0;
    for is = 1:length(s)
       cur_ptn = ptn(s(is),:);
       for iaff = 1:nAfferents
           if ~isempty(cur_ptn{1,iaff})
               spliced_ptn{1,iaff} = [spliced_ptn{1,iaff},cur_ptn{1,iaff}+ T_start];
           end
       end
       T_start = T_start + get_Tmax2(cur_ptn)+interval;
    end
    Tmax = T_start - interval;
end
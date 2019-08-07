% return the vector recording the max spike timing of each spike pattern in 'ptn' 
function Tmax = get_Tmax(ptn)
 [nPtns,nAfferents] = size(ptn);
 Tmax = zeros(nPtns,1);
 
 for iptn = 1:nPtns
     tmp = zeros(1,nAfferents);
     for iaff = 1:nAfferents
         if ~isempty(ptn{iptn,iaff})
             tmp(iaff) = max(ptn{iptn,iaff});
         end
     end
     Tmax(iptn) = max(tmp);
 end
end
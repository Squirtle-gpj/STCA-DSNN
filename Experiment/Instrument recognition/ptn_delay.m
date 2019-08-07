function new_ptn = ptn_delay(ptn,k,t_delay)
%ptn:spike pattern 
% k: terminal ����
% �ӳ�ʱ��
[nPtns,nAfferents] = size(ptn);
new_ptn = cell(nPtns,nAfferents*k);
for iptn = 1:nPtns
    for iaff = 1:nAfferents
          if ~isempty(ptn{iptn,iaff})
                for ik = 1:k
                    new_ptn{iptn,(iaff-1)*k + ik} = ptn{iptn,iaff} + (ik-1)*t_delay;
                end
          end
    end
end
end
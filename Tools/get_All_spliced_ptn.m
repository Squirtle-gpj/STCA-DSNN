% randomly get several spliced patterns from 'trainData' and construct a
% new Data-struct to record this spliced_patterns. In addition,
% Data.actNumList records the labels of independent patterns in the spliced
% pattern. e.g., If the ith spliced pattern contains 5 independent patterns
% which labels are 1,2,3,4,5 respectively, Data.actNumList(:,i) = [1;2;3;4;5]
% nTrials: the total number of spliced patterns
% interval: time interval between 2 ajacent independent patterns, unit:
% second
% nPtns_per_trial: the number of independent patterns in each spliced
% pattern
% %selected_labels:  the indexes of the chosen labels, e.g., [1,2,3]
% indicates choosing data of the first three classes
function Data = get_All_spliced_ptn(trainData,nTrials,interval,nPtns_per_trial,selected_labels)
   nLabels = length(selected_labels);
   for ilabel = 1:nLabels
       tmp = find(trainData.Labels == ilabel);
       eaStart(ilabel) = tmp(1) - 1;
       eaNptns(ilabel) = length(tmp);
   end
   Data.Labels_name = trainData.Labels_name(selected_labels);
   for itrial = 1:nTrials
       s = [];
       cur_actNumList = [];
       for iptn = 1:nPtns_per_trial
           curLabel = randperm(nLabels,1);
           curOffset = eaStart(curLabel) + randperm(eaNptns(curLabel),1);
           s = [s,curOffset];
           cur_actNumList = [cur_actNumList;curLabel];
       end
       [curptn, curTmax] = splice_ptn(trainData.ptn,s,interval);
       Data.ptn(itrial,:) = curptn;
       Data.Tmax(itrial) = curTmax;
       Data.actNumList(:,itrial) = cur_actNumList;
       disp(itrial);
   end
   
end
function [TrainData,ValData] = divideData(Data,ratio)
     nPtns = length(Data.ptn);
     nCls = length(Data.Labels_name);
     TrainIdx = [];
     ValIdx = [];
     for icls = 1:nCls
         tmp = find(Data.Labels == icls);
         tmp = tmp(randperm(length(tmp)));
         divide_point = ceil((ratio(2)/(ratio(1) + ratio(2)))*length(tmp));
         ValIdx = [ValIdx,tmp(1:divide_point)];
         TrainIdx = [TrainIdx,tmp(divide_point+1:end)];
     end
     TrainData.Labels_name = Data.Labels_name;
     TrainData.ptn = Data.ptn(TrainIdx,:);
     TrainData.Tmax = Data.Tmax(TrainIdx);
     TrainData.Labels = Data.Labels(TrainIdx);
     
     ValData.Labels_name = Data.Labels_name;
     ValData.ptn = Data.ptn(ValIdx,:);
     ValData.Tmax = Data.Tmax(ValIdx);
     ValData.Labels = Data.Labels(ValIdx);
 
end
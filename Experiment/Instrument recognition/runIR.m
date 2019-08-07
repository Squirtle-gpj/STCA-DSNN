load('M10_Data_rank3.mat');
Structure = MultilayerTempotronTr(TrainData,TestData,60,1,[700],1,0,0,0,[30e-3,30e-3]);
save(['M10_',num2str(Structure.best_result.accuracy),'.mat'],'Structure');

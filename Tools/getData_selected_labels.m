% getting a new struct only containing data of 'selected_labels' in 'OData'
% selected_labels:  the indexes of the chosen labels, e.g., [1,2,3]
% indicates choosing data of the first three classes
function Data = getData_selected_labels(OData,selected_labels)
    nLabels = length(selected_labels);
    Data.Labels_name = OData.Labels_name(selected_labels);
    start = 1;
    for ilabel = 1:nLabels
        curidx = find(OData.Labels == ilabel);
        nPtns = length(curidx);
        Data.ptn(start:start+nPtns-1,:) = OData.ptn(curidx,:);
        Data.Tmax(1,start:start+nPtns-1) = OData.Tmax(curidx);
        Data.Labels(1,start:start+nPtns-1) = OData.Labels(curidx);
    end
end
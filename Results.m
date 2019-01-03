clear all;
close all;
clc
errData = load('ModelTraining/test.out');

epoch = errData(:,1);
train_loss = errData(:,2:6:end);
val_loss = errData(:,3:6:end);
val_err = errData(:,4:6:end);
test_loss = errData(:,5:6:end);
test_err = errData(:,6:6:end);

legend_label = {'No Quantization','binary', '2bit quantization','3bit quantization','4bit quantization'};
smoothval = 20;
figure
for i = 1:5
    avg = movmean(train_loss(:,i),smoothval);
    avg = lowest(avg);
    plot(epoch,avg);
    hold on
end
title('Training Loss')
ylabel('Loss')
xlabel('Epoch')
set(gcf,'color','w')
legend(legend_label)
figure
for i = 1:5
    avg = movmean(val_loss(:,i),smoothval);
    avg = lowest(avg);
    plot(epoch,avg);
    hold on
end
title('Validation Loss')
ylabel('Error')
xlabel('Epoch')
set(gcf,'color','w')
legend(legend_label)
figure
for i = 1:5
    avg = movmean(val_err(:,i),smoothval);
    avg = lowest(avg);
    plot(epoch,avg);
    hold on
end
ylabel('Error')
xlabel('Epoch')
set(gcf,'color','w')
legend(legend_label)
title('Validation Error')
figure
for i = 1:5
    avg = movmean(test_loss(:,i),smoothval);
    avg = lowest(avg);
    plot(epoch,avg);
    hold on
end
ylabel('Loss')
xlabel('Epoch')
set(gcf,'color','w')
legend(legend_label)
title('Testing Loss')
figure
for i = 1:5
    avg = movmean(test_err(:,i),smoothval);
    avg = lowest(avg);
    plot(epoch,avg);
    hold on
end
ylabel('Error')
xlabel('Epoch')
set(gcf,'color','w')
title('Testing Error')
legend(legend_label)

quantize = @(x,q)((round(((x+1)/2)*(2^q-1))/(2^q-1))*2)-1;

files = {'ModelTraining/weights1.csv','ModelTraining/weights2.csv','ModelTraining/weights3.csv','ModelTraining/weights4.csv'};
file_idx = 1;
for file = files
     
    i = 1;
    fid = fopen(file{:});
    weight = cell(3,1)
    while (~feof(fid))
        line = cell2mat(strsplit(fgetl(fid),','))
        line_formatted = strrep(strrep(strrep(line,'"',''),']',''),'[','');
        temp_mat = cellfun(@(x)str2double(x),strsplit(line_formatted,' '));
        temp_mat(isnan(temp_mat)) = [];
        if strfind(line,'[')
            tempweight = temp_mat;
            
        else
            tempweight = [tempweight,temp_mat];
        end
        
        if strfind(line,']')
            weight{i} = tempweight
            i = i + 1;
        end
        
    end
    input2hiddenWeights{file_idx} = quantize(cell2mat(weight(1:16,:)),file_idx);
    hiddenActivation{file_idx} = quantize(cell2mat(weight(17,:)),file_idx);
    hidden2outputWeights{file_idx} = quantize(cell2mat(weight(18:end-1,:)),file_idx);
    outputActivation{file_idx} = quantize(cell2mat(weight(end,:)),file_idx);
    file_idx = file_idx + 1;
end

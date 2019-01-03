fid = fopen('ClaveVectors_Firm-Teacher_Model.txt');
data = zeros(10801,20);
i = 1;
while(~feof(fid))
    line = strsplit(fgetl(fid));
    data(i,:) = cellfun(@(x)str2double(x),line(1:20));
i = i+1;
end
save('Data.mat' ,'data')
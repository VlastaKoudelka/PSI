%load PSI and PLA data - eyes closed
%% get file names from the tree to filter the data
path = 'D:\psilocybin EEG files raw\active\';

dirSubj = dir([path 'PSI*']);
nSubj = length(dirSubj);

for i = 1:nSubj
    fileNames{i} = dir([path dirSubj(i).name '\EC\*.d']);
end

%% load appropriate data from BVA export
path = 'D:\Psilocybin all data\export_záloha\';

nEpoch = length(fileNames{1});

for i = 1:nSubj
    for j = 1:length(fileNames{i}) 
%         dir([path fileNames{i}(j).name(1:20) '*.dat'])
        dir([path fileNames{i}(j).name(1:14) '*' fileNames{i}(j).name(15:20)  '*.dat'])
    end
end
        
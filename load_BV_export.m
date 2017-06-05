%load PSI and PLA data - eyes closed
%% get file names from the tree to filter the data
path = 'D:\psilocybin EEG files raw\active\';

dirSubj = dir([path 'PSI*']);
nSubj = length(dirSubj);

for i = 1:nSubj
    fileNames{i} = dir([path dirSubj(i).name '\EC\*.d']);
end

%% load appropriate data from BVA export
path = 'D:\Vlasta\PSI_OZ_EDF\';

nEpoch = length(fileNames{1});

for i = 1:nSubj
    for j = 1:length(fileNames{i}) 
%         dir([path fileNames{i}(j).name(1:20) '*.dat'])
       filea = dir([path fileNames{i}(j).name(1:14) '*' fileNames{i}(j).name(15:20)  '*embed.edf']);
       fileb = dir([path fileNames{i}(j).name(1:14) '*' fileNames{i}(j).name(15:20)  '*long.edf']);
       
       if length(fileb)>0   %If a long version exist
           filea = fileb;   %use the long version
       end
       
       if length(filea)>0   %This will be probably always true
           SubFile(j) = filea;
           clear filea
       else
           clear filea
       end                    
    end
    GroupFile{i}=SubFile;
    clear SubFile;
end
        
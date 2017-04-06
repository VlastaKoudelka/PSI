%% t-SNE Human
clear all; close all
load D:\PSI\Psilo_coherence\tSNEHumanInputData

perplexity = 4;
nod = 12;                    %a number of initial dims
nocl = 2;
nop = size(SelLabels, 1);
%% Mean coherences and time differences

%calculate average values for all three recording intervals
PreAvg = mean(ValuesPre,3);     
Post1Avg = mean(ValuesPost1,3);
Post2Avg = mean(ValuesPost2,3);


DiffMat = [Post1Avg - PreAvg, Post2Avg - Post1Avg];



%% Dimmension reduction, clustering and silhouette

%dimmension reduction by stochastic neighbour embedding
mappedxo = tsne(DiffMat, [], 2, nod, perplexity);  %perplexity 5

%clustiring by k-means
IDX = kmeans(mappedxo,nocl);

%no of the clusters criterion
% noe = 10;
% for i = 1:noe
%     mappedx = tsne(DiffMat(randperm(nop),:), [], 2, nod, perplexity);  %perplexity 5
%     eva{i} = evalclusters(mappedx,'kmeans','silhouette','KList',[1:64]);
% end
%% Visualisation

%a difference matrix
figure(1)
imagesc(DiffMat)

%show t-SNE map
cmap = colormap(jet(128));
figure(2)
subplot(1,2,1)
scatter(mappedxo(:,1),mappedxo(:,2),[],cmap(round(IDX/max(IDX)*length(cmap)),:),'fill')
title('Psilocybin Clusters')



for i = 1:size(mappedxo,1)
    text(mappedxo(i,1),mappedxo(i,2),[SelLabels{i,1} ',' SelLabels{i,2}])
end

%map channel locs

chanlocs = readlocs('SRC\\10-20-tsne.loc');
clear ds
for i = 1:length(SelLabels) %get indices to obtain channel pairs
    for j = 1:length(chanlocs)
        if strcmp(chanlocs(j).labels, SelLabels{i,1})   %compare loc file
            ds.chanPairs(i,1) = j;                      %with selected elec
        elseif strcmp(chanlocs(j).labels, SelLabels{i,2})
            ds.chanPairs(i,2) = j;
        end
    end
end

ds.connectStrength = IDX;
ds.connectStrengthLimits = [0,2];

subplot(1,2,2)
colormap(jet(128));
topoplot_connect(ds, chanlocs);
title('Psilocybin Clusters Topography');
print -djpeg -r300 result.jpeg

%clusters criterion
% figure(3)
% for i = 1:length(eva)
%     plot(eva{i}.CriterionValues,'r','linewidth',2)
%     hold on
% end
% title('Silhouette Criterion')
% xlabel('a number of clusters')
% ylabel('criterion value')
% 
% print -djpeg -r300 criterion.jpeg




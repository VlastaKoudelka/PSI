function graph_dynamics

% close all

% [name,path] = uigetfile('*.mat','Open the source file');
% load ([path name], 'CoH');

load('D:\Filip_PSI_mysi\M Data\psi_coherences.mat', 'CoH')
load el_names electrodes
load con_loc x y
brain = imread('Mozek.png');

delta = CoH(:,1:4);
theta = CoH(:,5:8);
alpha = CoH(:,9:12);
beta = CoH(:,13:16);
h_beta = CoH(:,17:20);
gamma = CoH(:,21:24);
% h_gamma = CoH(:,25:28);

nocl = 6;                    %a number of the clusters
perplexity = 5;
nod = 28;                    %a number of initial dims
%% Calculations

%differences encode the trends
trend = diff(delta,1,2);
trend(:,4:6) = diff(theta,1,2);
trend(:,7:9) = diff(alpha,1,2);
trend(:,10:12) = diff(beta,1,2);
trend(:,13:15) = diff(h_beta,1,2);
trend(:,16:18) = diff(gamma,1,2);
% trend(:,19:21) = diff(h_gamma,1,2);

nod = size(trend,2)
% trend = diff(h_gamma,1,2);

%fingerprints of the trends
figure(1)
group = [ones(1,15) 2*ones(1,15) 3*ones(1,6)];
andrewsplot(trend,'Group',group,'linewidth',3)

%dimmension reduction by stochastic neighbour embedding
mappedx = tsne(trend, [], 2, nod, perplexity);  %perplexity 5

%clustiring by k-means
IDX = kmeans(mappedx,nocl);

%no of the clusters criterion
eva = evalclusters(mappedx,'kmeans','silhouette','KList',[1:36])
figure(2)
plot(eva.CriterionValues,'r','linewidth',2)
hold on

%time evolution of each cluster (averaged over frequencies)
mf_trend = (diff(delta,1,2) + diff(theta,1,2) ...
           + diff(alpha,1,2) + diff(beta,1,2))/4;

%% Visualization
cmap = colormap(hsv(nocl));

%visualize clusters
figure(3)
subplot(1,2,1)
scatter(mappedx(:,1),mappedx(:,2),[],cmap(IDX,:),'fill')

%show names
for i = 1:size(mappedx,1)
    text(mappedx(i,1),mappedx(i,2),electrodes(i,:))
end

%show clustered connections (different colors) in topo map
subplot(1,2,2) 
imagesc(brain)
for i = 1:length(IDX)
    line([x(2*i - 1) x(2*i)],[y(2*i - 1) y(2*i)],'Color',cmap(IDX(i),:),...
        'linewidth',2)
end

% %show clustered adrews plot
% subplot(2,2,3) 
% andrewsplot(trend,'Group',IDX,'linewidth',2)
% 
% %connection trends
% for i = 1:size(trend,1)
%     subplot(2,2,4)    
%     plot(mf_trend(i,:),'color',cmap(IDX(i),:),'linewidth',2)
%     hold on
% end

end
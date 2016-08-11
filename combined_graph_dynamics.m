function combined_graph_dynamics

close all

% [name,path] = uigetfile('*.mat','Open the source file');
% load ([path name], 'CoH');

load('D:\Filip_PSI_mysi\M Data\psi_coherences.mat', 'CoH')
CoH1 = CoH;                 %the first substance
load('D:\Filip_PSI_mysi\M Data\clo_coherences.mat', 'CoH')
CoH2 = CoH;                 %the second substance
load el_names electrodes
load con_loc x y
brain = imread('brain.png');

delta1 = CoH1(:,1:4);       %the first substance
theta1 = CoH1(:,5:8);
alpha1 = CoH1(:,9:12);
beta1 = CoH1(:,13:16);
h_beta1 = CoH1(:,17:20);
gamma1 = CoH1(:,21:24);
h_gamma1 = CoH1(:,25:28);

delta2 = CoH2(:,1:4);       %the second substance
theta2 = CoH2(:,5:8);
alpha2 = CoH2(:,9:12);
beta2 = CoH2(:,13:16);
h_beta2 = CoH2(:,17:20);
gamma2 = CoH2(:,21:24);
h_gamma2 = CoH2(:,25:28);

nocl = 4;                   %a number of the clusters
perplexity = 5;
nod = 5;                    %a number of initial dims
%% Calculations

%differences encode the trends
trend1 = diff(delta1,1,2);          %the first substance
trend1(:,4:6) = diff(theta1,1,2);
trend1(:,7:9) = diff(alpha1,1,2);
trend1(:,10:12) = diff(beta1,1,2);
trend1(:,13:15) = diff(h_beta1,1,2);
trend1(:,16:18) = diff(gamma1,1,2);
trend1(:,19:21) = diff(h_gamma1,1,2);

trend2 = diff(delta2,1,2);          %the second substance
trend2(:,4:6) = diff(theta2,1,2);
trend2(:,7:9) = diff(alpha2,1,2);
trend2(:,10:12) = diff(beta2,1,2);
trend2(:,13:15) = diff(h_beta2,1,2);
trend2(:,16:18) = diff(gamma2,1,2);
trend2(:,19:21) = diff(h_gamma2,1,2);

trend = [trend2 trend1];

% nod = size(trend,2)
% trend = diff(h_gamma,1,2);

%fingerprints of the trends
figure
group = [ones(1,15) 2*ones(1,15) 3*ones(1,6)];
andrewsplot(trend,'Group',group,'linewidth',3)

%dimmension reduction by stochastic neighbour embedding
mappedx = tsne(trend, [], 2, nod, perplexity);  %perplexity 5

%clustiring by k-means
IDX = kmeans(mappedx,nocl);

%time evolution of each cluster (averaged over frequencies)
% mf_trend = (diff(delta,1,2) + diff(theta,1,2) ...
%            + diff(alpha,1,2) + diff(beta,1,2))/4;

%% Visualization
cmap = colormap(hsv(nocl));

%visualize clusters
figure
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
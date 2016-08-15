load pair_coord_source x y pairs labels
load el_names electrodes

%% all electrodes
for i = 1:size(pairs,1)
    for j = 1:length(labels)
        for k = 1:length(labels)
            str = [labels{j},'-',labels{k}];
            if strcmp(str,pairs{i})
                x_pair_all(i,1) = x(j);
                x_pair_all(i,2) = x(k);
                y_pair_all(i,1) = y(j);
                y_pair_all(i,2) = y(k);
            end
        end
    end
end
% 
% brain = imread('brain2.png');
% imagesc(brain)
% 
% for i = 1:length(pairs)
%     line(x_paircoord(i,:),y_paircoord(i,:))
% end

%% selected electrodes

for i = 1:size(electrodes,1)
    for j = 1:length(labels)
        for k = 1:length(labels)
            str = [labels{j},' ',labels{k}];
            if strcmp(str,electrodes(i,:))
                x_pair_sel(i,1) = x(j);
                x_pair_sel(i,2) = x(k);
                y_pair_sel(i,1) = y(j);
                y_pair_sel(i,2) = y(k);
            end
        end
    end
end


for i = 1:size(pairs,1)
    for j = 1:size(electrodes,1)
        str = [electrodes(j,1:2),'-',electrodes(j,4:5)];
        if strcmp(str,pairs{i})
            index(j) = i - 1;
        end
    end
end       

brain = imread('brain2.png');
imagesc(brain)

for i = 1:length(electrodes)
    line(x_pair_sel(i,:),y_pair_sel(i,:))
end

save Locations x_pair_all x_pair_sel y_pair_all y_pair_sel index
       
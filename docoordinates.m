
for i = 1:size(pairs,1)
    for j = 1:length(labels)
        for k = 1:length(labels)
            str = [labels{j},'-',labels{k}]
            if strcmp(str,pairs{i})
                x_paircoord(i,1) = x(j);
                x_paircoord(i,2) = x(k);
                y_paircoord(i,1) = y(j);
                y_paircoord(i,2) = y(k);
            end
        end
    end
end

brain = imread('brain2.png');
imagesc(brain)

for i = 1:length(pairs)
    line(x_paircoord(i,:),y_paircoord(i,:))
end
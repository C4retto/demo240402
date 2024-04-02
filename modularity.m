function [ y ] = modularity( adj_mat, clu_assignment )
%% n: the number of clusters
n = max(clu_assignment);
%% L: the total links in the network
L = sum(sum(adj_mat))/2;
%% 
Q = 0;
for i = 1:n
    index = find(clu_assignment == i);
    S = adj_mat(index,index);
    ls = sum(sum(S))/2;
    ds = 0;
    for j = 1:length(index) %
        ds = ds + sum(adj_mat(index(j),:));
    end
    Q = Q + (ls/L - (ds/(2*L))^2);   
end
y = -Q;
end


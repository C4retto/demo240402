function labels = Decode(locus)
    N = length(locus);
    current_cluster = 1;
    cluster_assignment = zeros(1, N);
    for i = 1:N
        cluster_assignment(i) = -1;
    end
    for i = 1:N
        ctr = 1;
        if (cluster_assignment(i) == -1)
            previous = zeros(1,N);
            cluster_assignment(i) = current_cluster;
            neighbour = locus(i);
            previous(ctr) = i;
            ctr = ctr + 1;
            while (cluster_assignment(neighbour) == -1)
                previous(ctr) = neighbour;
                cluster_assignment(neighbour) = current_cluster;
                neighbour = locus(neighbour);
                ctr = ctr + 1;
            end
            if (cluster_assignment(neighbour) ~= current_cluster)
                ctr = ctr - 1;
                while (ctr >= 1)
                    cluster_assignment(previous(ctr)) = cluster_assignment(neighbour);
                    ctr = ctr - 1;
                end
            else
                current_cluster = current_cluster + 1;
            end  
        end
    end
    labels = cluster_assignment;
end
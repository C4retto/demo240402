function locus = Decode1(sparsem, numVar, indi)
    rvc = zeros(2, numVar);
    for j = 1: numVar
        adjr(j, :) = indi .* sparsem(j, :);
        rmaxp = max(adjr(j,:));
        rvcindex = find(adjr(j,:)==rmaxp);
        if length(rvcindex) == 1
            rvc(1,j)=j;
            rvc(2,j)=rvcindex;
        else
            rvc(1,j) = j;
            rvc(2,j) = randsample(rvcindex,1);
        end
    end
    locus = rvc(2, :);
end
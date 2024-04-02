function matrix = Adjreverse(EDGE, numVar)
    matrix = zeros(numVar, numVar, 'logical');
    for i = 1 : numVar
        index = find(EDGE(:,1)==i);
        matrix(i, EDGE(index, 2)) = 1;
        matrix(EDGE(index, 2), i) = 1;
    end
end
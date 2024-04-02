function [pop,objvals] = ncmoeainit(p, NetEmb, popsize, param)
	% SimMatrix = Tosim_matrix(p.adj,1);
	cen = Centrality(p.adj, p.degree);
	EmbDist = squareform(pdist(NetEmb));
	[~,pnode1] = max(cen);
	
	classlabel = zeros(1, p.numVar);
	[~, farnode1] = max(EmbDist(pnode1, :));
	idx1 = kmeans(NetEmb, 2, 'Start', NetEmb([pnode1, farnode1], :));
	classlabel(idx1==idx1(pnode1)) = 1;
	classlabel(classlabel==0) = 2;
	
	condCenterFlag = false(1, p.numVar);
	condCenter = find(classlabel == 1);
	condCenterFlag(condCenter) = true;
	
	for i = 1:popsize
	% main init.
		indi = rand(1,p.numVar);
		indi(~condCenterFlag) = indi(~condCenterFlag)*param;
		pop(i,:) = Decode1(p.sparsem, p.numVar, indi);
	end
    objvals = p.func(pop, popsize);
end
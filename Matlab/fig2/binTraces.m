% Takes a trace sampled at some higher resolution and reduces it by
% averaging over non-overlapping windows of size 'binWidth'
% Trace should be one-dimensional and have length divisible by binWidth

function result=binTraces(inpTraces,binWidth)

result=mean(reshape(inpTraces,binWidth,[]));


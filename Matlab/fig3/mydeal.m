%original code from Quentin Huys

function mydeal(S);

A=fieldnames(S);
for k=1:length(A)
	assignin('caller',A{k},S.(A{k}));
end

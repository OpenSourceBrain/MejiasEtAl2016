%ranks

function [z1,z2]=ranks2(mDAI,Nareas2)

%---------------
%%we compute the hierarchical positions:
mDAIp=5.*mDAI'; %rescale mDAI to -5:5, and transpose.
z1=zeros(Nareas2,Nareas2);
for i=1:Nareas2
    z1(i,:)=min(mDAIp(i,:)); %find smallest mDAI for source i,
end
mDAIp=mDAIp+ones(Nareas2,Nareas2)-z1; %and set it to one.

z1=zeros(1,Nareas2);z2=z1;
for i=1:Nareas2
    [~,~,z0]=find(mDAIp(:,i)); %consider only functionally connected pairs
    z1(1,i)=mean(z0);
    %z2(1,i)=std(z0);
    z2(1,i)=SEM_calc(z0);
end






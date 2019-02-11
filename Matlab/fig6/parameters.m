% parameters for the model:

function [par]=parameters(Areas,ROI,flnMat,slnMat,wiring)


Nareas=length(Areas);
% We put all the relevant parameters in the parameter structure "par":
%transduction function parameters are inside their corresponding functions.
%time constants, in seconds:
par.dt=0.2e-3;par.triallength=200.;par.transient=20.;
sc2=1.;sc5=5.;
par.tau=[0.006*sc2 0.015*sc2 0.006*sc5 0.015*sc5]; %re2 ri2 re5 ri5
par.tstep=((par.dt)./(par.tau))';
par.tstep=par.tstep*ones(1,Nareas);% to cover all areas
sig=[0.3 0.3 0.45 0.45];
par.tstep2=(((par.dt.*sig.*sig)./(par.tau)).^(0.5))';
par.tstep2=par.tstep2*ones(1,Nareas);% to cover all areas
par.binx=20;par.eta=0.2;G=1.1;
%eta=0 means recording only from L5, and eta=1 means recording only from
%L2/3. So eta (or 1-eta, to be exact) is a measure of the recording depth.

% local and interlaminar synaptic coupling strengths:
J2e=1.00;      % L2/3 excit to L5 excit
J2i=0.;       % L2/3 excit to L5 inhib
J5e=0.;       % L5 excit to L2/3 excit
J5i=0.75;      % L5 excit to L2/3 inhib

par.J=zeros(4,4);
%local, layer 2:
par.J(1,1)=1.5;par.J(1,2)=-3.25;
par.J(2,1)=3.5;par.J(2,2)=-2.5;
%local, layer 5:
par.J(3,3)=1.5;par.J(3,4)=-3.25;
par.J(4,3)=3.5;par.J(4,4)=-2.5;
%inter-laminar:
par.J(3,1)=J2e;par.J(4,1)=J2i;
par.J(1,3)=J5e;par.J(2,3)=J5i;

%background inputs (common to all areas)
par.inputbg=0.*ones(4,Nareas);


flnM=flnMat(Areas,Areas);
slnM=slnMat(Areas,Areas);

%range compression for the connection weights:
flnM=1.2*flnM.^0.30;

%selectivity matrix, s(post,pre):
par.s=0.1*ones(Nareas,Nareas);

%W(areapost,areapre), is specific for each local population.
par.Wff=(flnM.*slnM);
par.Wfb=(flnM.*(ones(Nareas,Nareas)-slnM));
%we transform the NxN distance matrix into NxN delay matrix:
wires=wiring(Areas,Areas);
par.delay=round(1+wires./(1500*par.dt)); %v=1.5m/s
%delays given now in units of dt=0.1ms



%we properly normalize the FLNs entering each area:
normff=sum(par.Wff,2);
normfb=sum(par.Wfb,2);
for i=1:Nareas
    par.Wff(i,:)=G.*(par.Wff(i,:)/normff(i,1));
    par.Wfb(i,:)=G.*(par.Wfb(i,:)/normfb(i,1));
end





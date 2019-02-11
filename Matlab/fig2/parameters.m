% parameters for the model:

function [par]=parameters(s)


% We put all the relevant parameters in the parameter structure "par":
%time constants, in seconds:
par.dt=0.2e-3;par.triallength=25.;par.transient=5.; %triallength=100
sc2=1.;%with 1, gamma freq is around 35 to 50 Hz
sc5=5.;%with 5, alpha freq is around 8 to 10 Hz
par.tau=[0.006*sc2 0.015*sc2 0.006*sc5 0.015*sc5]; %constants for re2 ri2 re5 ri5
par.tstep=((par.dt)./(par.tau))';
sig=0.3.*[1 1 1.5 1.5];
par.tstep2=(((par.dt.*sig.*sig)./(par.tau)).^(0.5))';



% local and interlaminar synaptic coupling strengths (layers are decoupled here):
J2e=0.5*0;      % L2/3 excit to L5 excit
J2i=0.;       % L2/3 excit to L5 inhib
J5e=0.;       % L5 excit to L2/3 excit
J5i=0.1*0;       % L5 excit to L2/3 inhib.

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

%background inputs:
par.inputbg=0.*ones(4,1); %re2 ri2 re5 ri5


%large-scale connections (we don't use them here)
par.W=zeros(2,2,4,2);
par.W(2,1,1,1)=1;      %V1 to V4, supra to supra
par.W(2,1,2,1)=0;      %V1 to V4, supra to infra
par.W(1,2,1,2)=s*2;    %V4 to V1, infra to supra excit
par.W(1,2,3,2)=0.5;    %other option, same result
par.W(1,2,2,2)=(1-s);  %V4 to V1, infra to infra excit
par.W(1,2,4,2)=0.5;    %other option, same result

















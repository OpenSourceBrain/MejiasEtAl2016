% Main code to generate functional hierarchies
%
% To obtain a functional hierarchy as in the SciAdv paper, run this code 5
% times with different values of 'indicedata' and average the results using
% the code 'rankfast.m'
%
% Jorge Mejias, 2016


function []=main(indicedata)


%add path for the GC toolbox:
%addpath('/scratch/jmp20/mvgc_v1.0');
%startup;%matlabpool('open', 'local', 10);

%initialize random number generator and load data:
seed1=round(938190+12*indicedata);rng(seed1);

%-----------Data to obtain from core-nets.org:

load('subgraphData30.mat'); %this must include:
%FLN (as flnMat), rank-ordered, 
%SLN (as SLNmat), rank-ordered,
%a list of area names rank-ordered (as areaList)
load('subgraphWiring30.mat'); 
%area-to area distances, rank-ordered and given in mm.

%------------------Now the simulations:
Areas=1:30;
Nareas=length(Areas);
ROI=[1 2 3 4 6 8 9 13]; %Kennedy-Fries Neuron 2015 selection
Nareas2=length(ROI);

%prepare the parameters and compute the main data:
par=parameters(Areas,ROI,flnMat,slnMat,wiring);
[mDAI,X2,X5,DAI,f,F,fshuf,Fshuf]=hierarchy(indicedata,Areas,par,ROI,areaList,flnMat,slnMat);

A=mDAI;
B=F;
C=f;
D=Fshuf;
E=fshuf;

%let's get the gamma and alpha power along the cortex:
gammac=zeros(Nareas,30);alphac=gammac;
for j=1:30
    for i=1:Nareas
        %gamma power:
        [~,~,fg,zg]=analysis(par,X2(i,:,j),30);
        %alpha power:
        [~,~,fa,za]=analysis(par,X5(i,:,j),3);
        gammac(i,j)=zg;alphac(i,j)=za;
    end
end


%save the data
output=sprintf('output%d.mat',indicedata);
save(output,'A','B','C','D','E','alphac','gammac','DAI','par');





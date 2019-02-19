%---------------------------------------%
%                                   	%
%              rankfast                 %
%                                       %
%---------------------------------------%

%first, we have manually introduce the number of trials:
trials=1:5;supestad=length(trials);

%let's couple all the output files together (assuming we run 5 hierarchies)
%this is to compile different trials together into one file
Am=cell(supestad,1);
Bm=Am;Cm=Am;Dm=Am;Em=Am;
for i=1:supestad
  name=sprintf('output%d.mat',i);
  load(name);Am{i}=A;Bm{i}=B;
  Cm{i}=C;Dm{i}=D;Em{i}=E;
end

%change notation:
clear A B C D E;
A=Am;B=Bm;C=Cm;D=Dm;E=Em;
save output.mat A B C D E alphac gammac DAI par;


%---------now the analyze the data-----------%

%load the multi-trial data:
%load 'output.mat';

Nareas=30;Nareas2=8;
load('subgraphData30.mat'); %FLN and SLN, rank-ordered.
load('subgraphWiring30.mat'); %distances rank-ordered and given in mm.
ROI=[1 2 3 4 6 8 9 13]; %Kennedy-Fries 2015 selection
chosen=0; %when zero, it plots the average over suptrials

% Mask to compute the hierarchy:
% We find the highest value of F among the no-links (i.e. pair with
% fln=0) and automatically set the threshold a 5% above that value.
% We use the resulting matrix for all trials, so that the comparison
% is fair.
nolinks=find(flnMat(ROI,ROI)==0);
totalGC=B{1};totalGC(1:Nareas2+1:Nareas2^2)=0; %first trial, for example.
umbral=1.05*max(totalGC(nolinks)); %automatic threshold
%setting a minimum value:
totalGC(totalGC<=umbral)=0;
adj=and(totalGC,ones(Nareas2,Nareas2));
adj(1:Nareas2+1:Nareas2^2)=1; %set diagonal to ones

hranks=zeros(supestad,Nareas2);
SEMranks=zeros(supestad,Nareas2);
for j=1:supestad
    i=trials(j);
    mDAI=zeros(Nareas2,Nareas2);F=mDAI; %set proper dimensions
    mDAI=A{i};F=B{i};
    [hranks(j,:),SEMranks(j,:)]=ranks2(mDAI,adj,Nareas2);
end


%------------------------------------------------------
%we build the final functional hierarchy:
z3=areaList(ROI);
if chosen==0
    z1=mean(hranks);
    z2=SEM_calc(hranks)';
else
    z1=hranks(chosen,:);z2=SEMranks(chosen,:);
end
[~,ind]=sort(z1); %sort areas
n1=z1(ind);n2=z2(ind);n3=z3(ind);

figure('Position',[100 100 700 500])
errorbar(1:Nareas2,n1,n2,'o','LineWidth',3);
set(gca,'XTick',1:Nareas2);
set(gca,'XTickLabel',n3);
ylim([1 7]);ylabel('Hierarchical level');


%-----------------------------------------------------

%to compute the DAIxSLN correlations:
for i=1:supestad
    f=C{i};
    [frequ,slndai(i,:),DAI]=slnxdai(par,ROI,f,slnMat);
end

figure('Position',[100 100 1100 400])
subplot(1,2,1)
baseline=zeros(1,length(frequ));
slndaimean=mean(slndai,1);
slndaisig=std(slndai,1);
myeb2(frequ,slndaimean,slndaisig,[0 0 1]);hold on;
plot(frequ,baseline,'k--','LineWidth',2);
xlim([0 100]);
set(gca,'FontSize',18,'LineWidth',3,'TickLength',[0.01 0.01]);
ylabel('DAI x SLN correlation');
xlabel('Frequency (Hz)');

%to calculate the mDAI x SLN correlation (only one trial is ok):
f=C{1};
[SLNchain2,mDAIchain2,rho2,pval2]=mdaimatrix(par,ROI,f,slnMat);
subplot(1,2,2)
plot(SLNchain2,mDAIchain2,'.','MarkerSize',20);
title(sprintf('R=%f      p-value=%f',rho2,pval2));
set(gca,'FontSize',18,'LineWidth',3,'TickLength',[0.01 0.01]);
ylim([-0.7 0.7]);ylabel('mDAI');xlabel('SLN');


%-----------------------------------------------

meangamma=mean(gammac,2);
siggamma=std(gammac');
meanalpha=mean(alphac,2);
sigalpha=std(alphac');

figure('Position',[100 100 1100 400])
subplot(1,2,1)
%myeb2(1:30,meanalpha',siggalpha,[0.99 0.45 0.1]);
myeb2(1:Nareas2,meanalpha(ROI)',sigalpha(ROI),[0.99 0.45 0.1]);
errorbar(1:Nareas2,meanalpha(ROI)',sigalpha(ROI),...
'o','Color',[0.99 0.45 0.1],'LineWidth',2);
set(gca,'XTick',1:Nareas2);
set(gca,'XTickLabel',z3);
ylabel('Alpha power');
subplot(1,2,2)
%myeb2(1:30,meangamma',siggamma,[0.3 0.7 0.3]);
myeb2(1:Nareas2,meangamma(ROI)',siggamma(ROI),[0.3 0.7 0.3]);
errorbar(1:Nareas2,meangamma(ROI)',siggamma(ROI),...
'o','Color',[0.3 0.7 0.3],'LineWidth',2);
set(gca,'XTick',1:Nareas2);
set(gca,'XTickLabel',z3);
ylabel('Gamma power');









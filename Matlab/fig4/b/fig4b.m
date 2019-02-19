%figure 4
%this is for FB



hfig=figure(1);
clf('Figure 1','reset')
set(hfig,'Position',[600,400,1500,1200]);
hold on;




j=1; % we plot spectra for V1(j=1) or V4(j=2)
filtro=100;
resbin2=20;
linew=4;
subplot(2,2,3);
pz0=(squeeze(px20(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz0,1)
    pxx0(i,:)=smooth(pz0(i,:),filtro);
end
pxx20=mean(pxx0,1);
pxx20sig=std(pxx0,1);
myeb2(fx(1:resbin2:end),pxx20(1:resbin2:end),...
    pxx20sig(1:resbin2:end),0.6*[0.3 0.7 0.3]);hold on;
%stim:
pz=(squeeze(px2(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz,1)
    pxx(i,:)=smooth(pz(i,:),filtro);
end
pxx2=mean(pxx,1);
pxx2sig=std(pxx,1);
myeb2(fx(1:resbin2:end),pxx2(1:resbin2:end),...
    pxx2sig(1:resbin2:end),[0.3 0.7 0.3]);hold on;
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
xlim([20,80]);
xlabel('Frequency (Hz)');
ylabel('V1 L2/3E power');



subplot(2,2,4)
resbin5=10;
%rest:
pz0=(squeeze(px50(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz0,1)
    pxx0(i,:)=smooth(pz0(i,:),filtro);
end
pxx50=mean(pxx0,1);
pxx50sig=std(pxx0,1);
myeb2(fx(1:resbin5:end),pxx50(1:resbin5:end),...
    pxx50sig(1:resbin5:end),0.6*[0.99 0.45 0.1]);hold on;
%stim:
pz=(squeeze(px5(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz,1)
    pxx(i,:)=smooth(pz(i,:),filtro);
end
pxx5=mean(pxx,1);
pxx5sig=std(pxx,1);
myeb2(fx(1:resbin5:end),pxx5(1:resbin5:end),...
    pxx5sig(1:resbin5:end),[0.99 0.45 0.1]);hold on;
%constant input:
pz9=(squeeze(px59(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz9,1)
    pxx9(i,:)=smooth(pz9(i,:),filtro);
end
pxx59=mean(pxx9,1);
pxx59sig=std(pxx9,1);
myeb2(fx(1:resbin5:end),pxx59(1:resbin5:end),...
    pxx59sig(1:resbin5:end),[0.2 0.2 0.9]);hold on;
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
xlim([5,20]);
xlabel('Frequency (Hz)');
ylabel('V1 L5/6E power');




%---------

powpeak0=powerpeak0;powpeak1=powerpeak1;
%rest:
barras0=zeros(4,3);
for i=1:4  %V1L2/3 - V1L5 - V4L2/3 - V4L5
  barras0(i,:)=[i; mean(powpeak0(i,:)); std(powpeak0(i,:))];
end
%stim:
barras1=zeros(4,3);
for i=1:4  %V1L2/3 - V1L5 - V4L2/3 - V4L5
  barras1(i,:)=[i; mean(powpeak1(i,:)); std(powpeak1(i,:))];
end
%bars:
barras=zeros(4,3);
barras1(:,1)=barras1(:,1)+0.5;
%the data we want to plot is:
j=1;k=j+1; %j=1 to record in V1, j=3 to record in V4.
barras(1,:)=barras0(j,:); %L2/3 rest
barras(2,:)=barras1(j,:); %L2/3 stim
barras(3,:)=barras0(k,:); %L5 rest
barras(4,:)=barras1(k,:); %L5 stim

%renaming:
barrasgamma=barras(1:2,:);
barrasalpha=barras(3:4,:);




%-----------------now, the insets:


sh1=subplot(2,2,1);
set(sh1,'Position',[0.34 0.33 0.12 0.12]);
bar(1,barrasgamma(1,2),'FaceColor',0.6*[0.3 0.7 0.3],'LineWidth',3);hold on;
bar(2,barrasgamma(2,2),'FaceColor',[0.3 0.7 0.3],'LineWidth',3);hold on;
errorbar([1 2],barrasgamma(:,2),barrasgamma(:,3),'.k','LineWidth',3);
set(gca,'FontSize',22,'LineWidth',5,'TickLength',[0.03 0.03],'YTick',[0 0.005]);
set(gca,'box','off');
if s<0.5 %significance (nonselect):
    z=0.008;z2=0.0005;
    plot([1,1,2,2],[z,z+z2,z+z2,z],'-k','LineWidth',3);
    text(1.5,z+z2*2.5,'***','FontSize',24,'horizontalalignment','center');
elseif s>=0.5 %significance (select):
    z=0.035;z2=0.002;
    plot([1,1,2.5,2.5],[z,z+z2,z+z2,z],'-k','LineWidth',3);
    text(1.5,z+z2*2.5,'***','FontSize',24,'horizontalalignment','center');
end
%the rest:
xlim([0.25,2.75]);
tope=max(barras(1,2),barras(2,2));
ylim([0,tope*1.75]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'Rest','Stim'})
ylabel('Gamma power');




sh2=subplot(2,2,2);
set(sh2,'Position',[0.78 0.33 0.12 0.12]);
bar(1,barrasalpha(1,2),'FaceColor',0.6*[0.99 0.45 0.1],'LineWidth',3);hold on;
bar(2,barrasalpha(2,2),'FaceColor',[0.99 0.45 0.1],'LineWidth',3);hold on;
errorbar([1 2],barrasalpha(:,2),barrasalpha(:,3),'.k','LineWidth',3);
set(gca,'FontSize',22,'LineWidth',5,'TickLength',[0.03 0.03],'YTick',[0 0.2]);
set(gca,'box','off');
if s<0.5 %significance (nonselect):
    z=0.18;z2=0.015;
    plot([1,1,2,2],[z,z+z2,z+z2,z],'-k','LineWidth',3);
    text(1.5,z+z2*2.5,'***','FontSize',24,'horizontalalignment','center');
elseif s>=0.5 %significance (select):
    z=0.16;z2=0.008;
    plot([1,1,2,2],[z,z+z2,z+z2,z],'-k','LineWidth',3);
    text(1.5,z+z2*3.5,'n.s.','FontSize',24,'horizontalalignment','center');
end
%the rest:
xlim([0.25,2.75]);
tope=max(barras(3,2),barras(4,2));
ylim([0,tope*1.75]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'Rest','Stim'})
ylabel('Alpha power');












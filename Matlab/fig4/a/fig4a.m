%figure 4
%this is for FF


format long;
hfig=figure(1);
clf('Figure 1','reset')
set(hfig,'Position',[600,400,1500,1200]);
hold on;

barrasgamma=zeros(2,2);
barrasalpha=zeros(2,2);


j=2; % we plot spectra for V1(j=1) or V4(j=2)
filtro=100;
resbin2=20;
linew=4;
subplot(2,2,3);
pz0=(squeeze(px20(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz0,1)
    pxx0(i,:)=smooth(pz0(i,:),filtro);
end
pxx20=mean(pxx0,1);pxx20sig=std(pxx0,1);
myeb2(fx(1:resbin2:end),pxx20(1:resbin2:end),...
    pxx20sig(1:resbin2:end),0.6*[0.3 0.7 0.3]);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1=pxx20(find(fx>20 & fx<80));
z2=pxx20sig(find(fx>20 & fx<80));
[b1,b2]=max(z1);
barrasgamma(1,1)=z1(b2);
barrasgamma(1,2)=z2(b2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1=pxx2(find(fx>20 & fx<80));
z2=pxx2sig(find(fx>20 & fx<80));
[b1,b2]=max(z1);
barrasgamma(2,1)=z1(b2);
barrasgamma(2,2)=z2(b2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%loglog(fx,pxx2);hold on;
pz9=(squeeze(px29(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz9,1)
    pxx9(i,:)=smooth(pz9(i,:),filtro);
end
pxx29=mean(pxx9,1);
pxx29sig=std(pxx9,1);
myeb2(fx(1:resbin2:end),pxx29(1:resbin2:end),...
    pxx29sig(1:resbin2:end),[0.2 0.2 0.9]);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
xlim([20,80]);
xlabel('Frequency (Hz)');
ylabel('V4 L2/3E power');



subplot(2,2,4)
resbin5=10;
pz0=(squeeze(px50(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz0,1)
    pxx0(i,:)=smooth(pz0(i,:),filtro);
end
pxx50=mean(pxx0,1);
pxx50sig=std(pxx0,1);
myeb2(fx(1:resbin5:end),pxx50(1:resbin5:end),...
    pxx50sig(1:resbin5:end),0.6*[0.99 0.45 0.1]);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1=pxx50(find(fx>5 & fx<20));
z2=pxx50sig(find(fx>5 & fx<20));
[b1,b2]=max(z1);
barrasalpha(1,1)=z1(b2);
barrasalpha(1,2)=z2(b2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stim:
pz=(squeeze(px5(:,j,:)))';
fx=(fx2(:,j,1))';
for i=1:size(pz,1)
    pxx(i,:)=smooth(pz(i,:),filtro);
end
pxx5=mean(pxx,1);
pxx5sig=std(pxx,1);
myeb2(fx(1:resbin5:end),pxx5(1:resbin5:end),...
    pxx5sig(1:resbin5:end),[0.99 0.45 0.1]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1=pxx5(find(fx>5 & fx<20));
z2=pxx5sig(find(fx>5 & fx<20));
[b1,b2]=max(z1);
barrasalpha(2,1)=z1(b2);
barrasalpha(2,2)=z2(b2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%loglog(fx,pxx5,'*');hold on;loglog(fx,fx.^(-2),'k','LineWidth',3);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
xlim([5,20]);
xlabel('Frequency (Hz)');
ylabel('V4 L5/6E power');



%-----------------now, the insets:

sh1=subplot(2,2,1);
set(sh1,'Position',[0.34 0.33 0.12 0.12]);
bar(1,barrasgamma(1,1),'FaceColor',0.6*[0.3 0.7 0.3],'LineWidth',3);hold on;
bar(2,barrasgamma(2,1),'FaceColor',[0.3 0.7 0.3],'LineWidth',3);hold on;
errorbar([1 2],barrasgamma(:,1),barrasgamma(:,2),'.k','LineWidth',3);
set(gca,'FontSize',22,'LineWidth',5,'TickLength',[0.03 0.03]);
set(gca,'box','off');xlim([0.25,2.75]);
%significance:
z=0.005;z2=0.0005;
plot([1,1,2,2],[z,z+z2,z+z2,z],'-k','LineWidth',3);
text(1.5,z+0.0005,'***','FontSize',24,'horizontalalignment','center');
tope=max(barrasgamma(1,1),barrasgamma(2,1));
ylim([0,tope*1.75]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'Rest','Stim'})
ylabel('Gamma power');


sh2=subplot(2,2,2);
set(sh2,'Position',[0.78 0.33 0.12 0.12]);
bar(1,barrasalpha(1,1),'FaceColor',0.6*[0.99 0.45 0.1],'LineWidth',3);hold on;
bar(2,barrasalpha(2,1),'FaceColor',[0.99 0.45 0.1],'LineWidth',3);hold on;
errorbar([1 2],barrasalpha(:,1),barrasalpha(:,2),'.k','LineWidth',3);
set(gca,'FontSize',22,'LineWidth',5,'TickLength',[0.03 0.03]);
set(gca,'box','off');xlim([0.25,2.75]);
%significance:
z=0.03;z2=0.003;
plot([1,1,2,2],[z,z+z2,z+z2,z],'-k','LineWidth',3);
text(1.5,z+0.005,'n.s.','FontSize',20,'horizontalalignment','center');
tope=max(barrasalpha(1,1),barrasalpha(2,1));
ylim([0,tope*1.75]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'Rest','Stim'})
ylabel('Alpha power');















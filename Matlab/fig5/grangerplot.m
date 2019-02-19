%plot
figure('Position',[600,400,1200,1000]);
hold on;

resbin=20;

%coherence plot:
subplot(2,2,1)
cohp=mean(pcoh0,1);
cohpsig=std(pcoh0,1);
cohf=mean(fcoh0,1);
myeb2(cohf,cohp,cohpsig,[0 0 1])
%set(gca,'FontSize',18,'LineWidth',3,'TickLength',[0.01 0.01]);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03]);
set(gca,'box','off');
xlabel('Frequency (Hz)');
ylabel('Coherence');
xlim([0 80]);
zc=1.1*max(cohp+cohpsig);
ylim([0 zc]);

%GC plot:
dt=par.binx*par.dt;
z2to1=squeeze(f0(1,2,:));
z1to2=squeeze(f0(2,1,:));
frequ0=1:1:length(z1to2);
nyq=2*length(z1to2)*dt;
frequ0=frequ0./nyq;
%resolution:
frequ=frequ0(1:resbin:end);
GC1to2=z1to2(1:resbin:end);
GC2to1=z2to1(1:resbin:end);
subplot(2,2,2)
plot(frequ,GC1to2,'Color',[0.3 0.7 0.3],'LineWidth',4);hold on;
plot(frequ,GC2to1,'Color',[0.99 0.45 0.1],'LineWidth',4);hold on;
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
legend('V1 to V4','V4 to V1');
xlim([0 80]);
zgc=1.1*max(max(z2to1),max(z1to2));
ylim([0 zgc]);
set(gca,'Layer','top');
ylabel('Granger causality');
xlabel('Frequency (Hz)');

%DAI:
perf0=permute(f0,[2 1 3]);
realDAI0=(f0-perf0)./(f0+perf0);
for i=1:Nareas2
  realDAI0(i,i,:)=0;
end
DAIplot0=squeeze(realDAI0(2,1,:));
%reduce the resolution:
DAIplot=DAIplot0(1:resbin:end);
subplot(2,2,3)
plot(frequ,DAIplot,'Color',[0.1 0.6 0.9],'LineWidth',4);hold on;
plot(frequ,zeros(length(frequ)),'k--','LineWidth',3);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03])
set(gca,'box','off');
xlim([0 80])
xlabel('Frequency (Hz)');
ylabel('DAI (from V1 to V4)');









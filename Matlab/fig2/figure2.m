%plot
figure('Position',[600,400,1000,800]);
hold on;

fx=mean(fxx,1);
fx=fx(fx<=100);
px=px(:,1:length(fx));
px2=px2(:,1:length(fx));

ppx(1,:)=px(1,:)-px(1,:);
ppx(2,:)=px(2,:)-px(1,:);sigx(2,:)=sqrt(px2(2,:).^2+px2(1,:).^2);
ppx(3,:)=px(3,:)-px(1,:);sigx(3,:)=sqrt(px2(3,:).^2+px2(1,:).^2);
ppx(4,:)=px(4,:)-px(1,:);sigx(4,:)=sqrt(px2(4,:).^2+px2(1,:).^2);


h1=myeb2(fx,ppx(4,:),sigx(4,:),1.0*[0.1 0.7 0.1]);hold on;
h2=myeb2(fx,ppx(3,:),sigx(3,:),0.4*[0.1 0.7 0.1]);hold on;
h3=myeb2(fx,ppx(2,:),sigx(2,:),0.05*[0.1 0.7 0.1]);hold on;
plot(fx,ppx(1,:),'--k','LineWidth',2);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03]);
lh=legend([h3 h2 h1],{'Input=2','Input=4','Input=6'});
set(lh, 'Position',[0.65, 0.65, .25, .25]);
xlim([10 80]);
ylim([0 0.003])
ylabel('Power (resp. rest)');
xlabel('Frequency (Hz)');


hgexport(gcf,'fig2b.eps', hgexport('factorystyle'), 'Format', 'eps');



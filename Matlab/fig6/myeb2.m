function myeb2(x,ymean,ysigma,color)
%color has to be on the format: color=[R G B]

revx=fliplr(x);
yminus=ymean-ysigma;
yplus=fliplr(ymean+ysigma);
fillerror=fill([x revx],[yminus yplus],0.25*color+[0.75 0.75 0.75]);
hold on;set(fillerror,'EdgeColor','None');
plot(x,ymean,'Color',color,'LineWidth',3);
set(gca, 'Layer','top');z=0.01;
hold off;



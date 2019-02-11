%plot, 8-areas


function c=grangerplot8big(indice,par,f,areaList,ROI)





%figure('Position',[700,600,705,605]);
hfig=figure(indice);
set(hfig,'Position',[100,100,1100,750]);
hold on;


k=1;
for i=1:length(ROI)
    
  for j=(i+1):length(ROI)
      
  dt=par.binx*par.dt;
  zbtoa=squeeze(f(i,j,:));
  zatob=squeeze(f(j,i,:));
  frequ=1:1:length(zatob);
  nyq=2*length(zatob)*dt;
  frequ=frequ./nyq;
  
  
  subplot(6,5,k)
  %subplot(3,4,k)
  plot(frequ,zatob,frequ,zbtoa,'LineWidth',3);
  
  stra=char(areaList(ROI(i)));
  strb=char(areaList(ROI(j)));
  stra2=sprintf('%s -> %s', stra,strb);
  strb2=sprintf('%s -> %s', strb,stra);
  hleg=legend(stra2,strb2);
  set(hleg,'FontSize',8);

  xlim([0 100]);
  a=max(zatob);b=max(zbtoa);c=max(a,b);
  limitey=max(c,1e-3);
  ylim([0 1.25*limitey]);
  ylabel('SPWCGC');
  xlabel('Frequency (Hz)');
  k=k+1;
  end
  
end


stri=sprintf('figu%d.jpeg',indice);
hgexport(gcf, stri, hgexport('factorystyle'), 'Format', 'jpeg');



c=1;

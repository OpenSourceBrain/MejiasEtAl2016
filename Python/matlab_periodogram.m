clear all
clc

% Load pickle file which contains the variables calculted calling the
% python function `interlaminar_analysis_periodogram`.
dic = loadpickle('../periodogram.pckl');

% Perform the same calculations as those defined on the Mejias code
numberofzones2 = size(dic.segment2,2);
tt=1;ff=1;pp=1;clear tt;clear ff;clear pp;
for i=1:numberofzones2
    [~,ff,tt,pp(:,:,i)]=spectrogram(dic.segment2(:,i),dic.window_len, dic.noverlap, ...
    dic.freq_displayed,dic.fs,'yaxis');
end

% Fnally we average (and apply log transformation for visualization):
figure()
gammawaves=mean(pp,3);
gtimemid=mean(tt);
gammatime=tt-gtimemid;
%subplot(3,2,[1 2 3 4])
surf(gammatime,ff,gammawaves);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03]);
set(gca, 'Layer','bottom');
axis xy;view(0,90);
ylabel('Frequency (Hz)');
xwin=0.24;xlim([-xwin xwin]);
xlim([-xwin xwin]);
shading interp;
colormap(jet);

% savefig
savefig('periodogram.fig')


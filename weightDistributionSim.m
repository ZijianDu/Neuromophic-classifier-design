
for Q = 1:4
close all;
SWDistrubtion = zeros(100000,1);
HWDistrubtion = zeros(100000,1);
for i = 1:100000
input = round(rand(16,1));
SWVals = linspace(-1,1,2^Q);
HWVals = -((2^Q)/2):((2^Q)/2-1);%0:(2^Q-1);%

idx = round(rand(16,1)*((2^Q)-1))+1;

SWDistrubtion(i) = input'*SWVals(idx)';
HWDistrubtion(i) = input'*HWVals(idx)';
end
figure
subplot(1,2,1)
histogram(SWDistrubtion,25,'Normalization','probability')
%hist(SWDistrubtion,length(unique(SWDistrubtion)))
title('Software X*W distribution')
ylabel('Probability')
xlabel('X*W')
subplot(1,2,2)
histogram(HWDistrubtion,'Normalization','probability')
ylabel('Probability')
xlabel('X*W')
%hist(HWDistrubtion,length(unique(HWDistrubtion)))
title('Hardware X*W distribution')
set(gcf,'color','w')
figure
x = -10:.001:10;
activation = sinh(x)./cosh(x);
subplot(5,1,1)
plot(x,activation,'lineWidth',3)
title('Continuous tanh activation')
titles = {'binary activation','2 bit activation','3 bit activation','4 bit activation'};
ylabel('f(x)')
xlabel('x')
% for Q = 1:4
% subplot(5,1,Q+1)
% plot(x,(round((2^Q-1)*((activation+1)/2))/((2^Q-1)/2))-1,'lineWidth',3)
% title(titles{Q})
% ylabel('f(x)')
% xlabel('x')
% end
% set(gcf,'color','w')

xVals = atanh(linspace(-1,1,2^Q+1));
xVals
round((xVals-mean(SWDistrubtion))*(var(HWDistrubtion)/var(SWDistrubtion))+ mean(HWDistrubtion))
end
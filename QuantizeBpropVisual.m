% 
% close all
% x = -1:.01:1;
% y = x.^2;
% y_b = round(x.^2*2)/2;
% 
% plot(x,y,'lineWidth', 3)
% hold on
% plot(x,y_b,'lineWidth', 3)
% set(gcf, 'color','w')
% title('Cost Function with Respect to Weights','fontSize',18)
% ylabel('Cost Function','fontSize',18)
% xlabel('Weight Value','fontSize',18)
% h = legend('Cost function with repsect to unquantized weights', 'Cost function with respect to quantized weights');
% set(h,'FontSize',16);
% xt = get(gca, 'XTick');
% set(gca, 'FontSize', 16)
% yt = get(gca, 'YTick');
% set(gca, 'FontSize', 16)

q = 2;
x = 0:.001:.999;
y = floor(x*2^q)/(2^q-1);
plot(x,y,'lineWidth',2)
xlabel('Unquantized Value','fontSize',16)
ylabel('Quantized Value','fontSize',16)
set(gcf,'color','w')
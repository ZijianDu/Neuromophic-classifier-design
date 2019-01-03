clear
a = linspace(-7,8,16);
for j = 1:1000
val2 = [];
for i = 1:1000;
   
   val2(i) = a(ceil(rand(16,1)*16))*a(ceil(rand(16,1)*16))';
    
end
c(i) = mean(val2);
end
mean(c)
 a= linspace(0,1,1000);
 b = round(a*3)/4;
 
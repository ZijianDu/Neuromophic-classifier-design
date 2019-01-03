function [b] = lowest(a)
lowestVal = inf;
for i = 1:length(a)
   if lowestVal>a(i)
        
       lowestVal = a(i);
   end
   b(i) = lowestVal;
end
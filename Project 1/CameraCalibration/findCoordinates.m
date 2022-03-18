function findCoordinates(name,filenameW,filenameC)

I = imread(name);
[imagePoints,boardSize] = detectCheckerboardPoints(I);
L = length(imagePoints);
imagePoints = int16(imagePoints);   % image points starts from origin and 
                                    % increase in Y direction means
figure(1);                                    % shorter axis of image
imshow(I);

hold on;
for i = 1:L
   plot(imagePoints(i,1),imagePoints(i,2),'+r', 'Linewidth', 2); 
end
xlabel('Y'); ylabel('X');   % Shorter axis of image is Y, Longer is X.
title('\bf Extracted Points from image');
xlswrite(filenameC,imagePoints);
Y = boardSize(1,1)-1;   
X = boardSize(1,2)-1;
worldCoordinate = [];
row = 1;
for x = 0:X-1
   for y = 0:Y-1
      worldCoordinate(row,1) = x;
      worldCoordinate(row,2) = y;
      worldCoordinate(row,3) = 0;
      row = row + 1;
   end
end
xlswrite(filenameW,worldCoordinate);
end
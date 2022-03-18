clc; clear; 

I=imread('retina1.jpg');
img(:,:)=I(:,:,2);
imshow(img);
[nr, nc] = size(img);

h = fspecial('gaussian',7,2);

x_gaussian = conv2(img,h,'same');

sqr = @(x) x.^2;

img = padarray(img,[1,1],0,'pre');
first_derivative_x = (sqr(((img(2:nr+1,2:nc+1) - (img(1:nr,2:nc+1))))) + (sqr((img(2:nr+1,2:nc+1)) - (img(2:nr+1,1:nc)))));

G1 = sqrt(double(first_derivative_x));

% Vertical Gradient
sobel_x = [[-1, 0, 1]; [-2, 0, 2]; [-1, 0, 1]];
sobel_y = [[1, 2, 1]; [0, 0, 0]; [-1, -2, -1]];

G_X = conv2(x_gaussian,sobel_x,'same');
G_Y = conv2(x_gaussian,sobel_y,'same');

G = sqrt((G_X.^2) + (G_Y.^2));
G2 = G1-G;

Theta = zeros(nr,nc);

% Computes the Edge Direction in degrees
for i=1:nr
    for j=1:nc
        Theta(i,j) = atand(G_Y(i,j) ./ G_X(i,j));
%         if Theta(i,j) < 0
%             Theta(i,j) = 360 + Theta(i,j);
%         end
    end
end

X_EDGE_DIRECTION = zeros(nr,nc);

angle = 22.5;
right_angle = 90;

for i=1:nr
    for j=1:nc
        if Theta(i,j) > right_angle*4-angle || Theta(i,j) >=0 && Theta(i,j) < angle || Theta(i,j) > right_angle*2-angle && Theta(i,j) <= right_angle*2+angle
            X_EDGE_DIRECTION(i,j) = 0;
        end
        
        if (Theta(i,j) >= angle && Theta(i,j) < right_angle-angle) || (Theta(i,j) > right_angle*2+angle && Theta(i,j) <= right_angle*3-angle)
            X_EDGE_DIRECTION(i,j) = 45;
        end
        
        if (Theta(i,j) >= right_angle-angle && Theta(i,j) < right_angle+angle) || (Theta(i,j) > right_angle*3-angle && Theta(i,j) <= right_angle*3+angle) 
            X_EDGE_DIRECTION(i,j) = 90;
        end 
        
        if (Theta(i,j) >= right_angle-angle && Theta(i,j) <= right_angle*2-angle) || (Theta(i,j) > right_angle*3+angle && Theta(i,j) <= right_angle*4-angle)
            X_EDGE_DIRECTION(i,j) = 135;
        end
    end
end

[nrtx, nctx] = size(X_EDGE_DIRECTION);
X_Mark = G;

G = padarray(G,[1,1],0,'both');

for i=2:nr+1
    for j=2:nc+1
        
        if X_EDGE_DIRECTION(i-1,j-1) == 0
            if (G(i,j) < G(i,j+1) || (G(i,j) < G(i,j-1)))
                X_Mark(i-1,j-1) = 0;
            end
        end
        
        if X_EDGE_DIRECTION(i-1,j-1) == 45
            if (G(i,j) < G(i-1,j+1)) || (G(i,j) < G(i+1,j-1))
                X_Mark(i-1,j-1) = 0;
            end
        end
        
        if X_EDGE_DIRECTION(i-1,j-1) == 90
            if ((G(i,j) < G(i-1,j)) || (G(i,j) < G(i+1,j)))
                X_Mark(i-1,j-1) = 0;
            end
        end
        
         if X_EDGE_DIRECTION(i-1,j-1) == 135
            if (G(i,j) < G(i-1,j-1)) || (G(i,j) < G(i+1,j+1))
                X_Mark(i-1,j-1) = 0;
            end
         end
    end
end

X_NMS = X_Mark; 

thresh_high = 0.1655;           
thresh_low = 0.13;              

X_Hyst = X_Mark;

X_Hyst_Pad = padarray(X_Hyst,[1 1],0,'both');

for i=2:nr+1
    for j=2:nc+1
        if X_Hyst_Pad(i,j) >= thresh_high
            X_Hyst(i-1,j-1) = 1;
        end
        if X_Hyst_Pad(i,j) < thresh_high && X_Hyst_Pad(i,j) >= thresh_low
            if X_Hyst_Pad(i,j+1) >= thresh_high || X_Hyst_Pad(i-1,j) >=thresh_high || X_Hyst_Pad(i,j-1) >= thresh_high || X_Hyst_Pad(i+1,j) >= thresh_high || X_Hyst_Pad(i+1,j+1) >=thresh_high || X_Hyst_Pad(i-1,j+1) >= thresh_high || X_Hyst_Pad(i-1,j-1) >= thresh_high || X_Hyst_Pad(i+1,j-1) >= thresh_high
                X_Hyst(i-1,j-1) = 1;
            else
                X_Hyst(i-1,j-1) = 0;
            end  
        end
        if X_Hyst_Pad(i,j) < thresh_low
            X_Hyst(i-1,j-1) = 0;
        end
    end
end

X_Final = X_Hyst;

figure,
imshow(X_Final);

[P, Pn] = bwlabel(X_Final, 8);

for i = 1: Pn
    [r, c] = find(P == i);
    [Rn, r2] = size(r);
    if Rn < 160
        P(r,c) = 0;
    end
end
figure,
imshow(P);

level = graythresh(P);
imBw = imbinarize(P,level);
BW2 = bwmorph(imBw, 'thin');
imshow(BW2);

ideal = edge(img,'canny');
figure,
imshow(ideal);


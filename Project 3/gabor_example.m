%function gf=gabor_feature(im,scale,orientation)

% Usage: EO = gaborconvolve(im,  nscale, norient, minWaveLength, mult, ...
%			    sigmaOnf, dThetaOnSigma)
%
% Arguments:
% The convolutions are done via the FFT.  Many of the parameters relate 
% to the specification of the filters in the frequency plane.  
%
%   Variable       Suggested   Description
%   name           value
%  ----------------------------------------------------------
%    im                        Image to be convolved.
%    nscale          = 4;      Number of wavelet scales.
%    norient         = 6;      Number of filter orientations.
%    minWaveLength   = 3;      Wavelength of smallest scale filter.
%    mult            = 2;      Scaling factor between successive filters.
%    sigmaOnf        = 0.65;   Ratio of the standard deviation of the
%                              Gaussian describing the log Gabor filter's transfer function 
%	                       in the frequency domain to the filter center frequency.
%    dThetaOnSigma   = 1.5;    Ratio of angular interval between filter orientations
%			       and the standard deviation of the angular Gaussian
%			       function used to construct filters in the
%                              freq. plane.
%
% Returns:
%
%   EO a 2D cell array of complex valued convolution results
%
%        EO{s,o} = convolution result for scale s and orientation o.
%        The real part is the result of convolving with the even
%        symmetric filter, the imaginary part is the result from
%        convolution with the odd symmetric filter.
%
%        Hence:
%        abs(EO{s,o}) returns the magnitude of the convolution over the
%                     image at scale s and orientation o.
%        angle(EO{s,o}) returns the phase angles.


clear; clc;
texture=imread('D:/PhD/OneDrive - Oklahoma A and M System/CM/Project 3/Samples/D8','bmp');
[X,Y]=size(texture);		% Size of the texture image
Ns=4; No=6;			% Numbers of scale and orientation in Gabor	
 
E0=gaborconvolve(texture,Ns,No,3,2,0.65,1.5);
 
for i=1:Ns
    for j=1:No
        ind=(i-1)*No+j;      		% Calculate the index of each sub-plot
        subplot(4,6,ind);           		% Create a multi-figure plot
        Mi=abs(E0{i,j});            		% Create the magnitude for each Gabor channel
        imshow(Mi,[]);              		% Show the Gabor filter output
        Miv{ind}=reshape(Mi,X*Y,1); 	% Reshape the matrix data to vector data
    end
end
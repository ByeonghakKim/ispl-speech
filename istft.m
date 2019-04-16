
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Inverse Short-Time Fourier Transform        %
%               with MATLAB Implementation             %
%                                                      %
% Author: Ph.D. Eng. Hristo Zhivomirov        12/26/13 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, t] = istft(STFT, awin, swin, hop, nfft, fs)
% function: [x, t] = istft(STFT, awin, swin, hop, nfft, fs)
%
% Input:
% stft - STFT-matrix (only unique points, time
%        across columns, frequency across rows)
% awin - analysis window function
% swin - synthesis window function
% hop - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
%
% Output:
% x - signal in the time domain
% t - time vector, s
% signal length estimation and preallocation
L = size(STFT, 2);          % determine the number of signal frames
wlen = length(swin);        % determine the length of the synthesis window
xlen = wlen + (L-1)*hop;    % estimate the length of the signal vector
x = zeros(1, xlen);         % preallocate the signal vector
% reconstruction of the whole spectrum
if rem(nfft, 2)             
    % odd nfft excludes Nyquist point
    X = [STFT; conj(flipud(STFT(2:end, :)))];
else                        
    % even nfft includes Nyquist point
    X = [STFT; conj(flipud(STFT(2:end-1, :)))];
end
% columnwise IFFT on the STFT-matrix
xw = real(ifft(X));
xw = xw(1:wlen, :);
% Weighted-OLA
for l = 1:L
    x(1+(l-1)*hop : wlen+(l-1)*hop) = x(1+(l-1)*hop : wlen+(l-1)*hop) + ...
                                      (xw(:, l).*swin)';
end
% scaling of the signal
W0 = sum(awin.*swin);                  
x = x.*hop/W0;                      
% generation of the time vector
t = (0:xlen-1)/fs;                 
end
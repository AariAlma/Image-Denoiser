function psf = motionPSF(len, angle)
%MOTIONPSF  Normalized motion-blur kernel.
%
%   psf = motionPSF(len, angle)
%
%   INPUTS
%     len   - number of pixels the blur spans
%     angle - direction of motion in degrees (0 = horizontal right)
%
%   OUTPUT
%     psf   - len x len double array, non-negative, sums to 1

psf    = zeros(len, len);
center = floor(len / 2) + 1;
cos_a  = cos(angle * pi / 180);
sin_a  = sin(angle * pi / 180);

for i = 1:len
    offset = i - center;
    col = round(center + offset * cos_a);
    row = round(center + offset * sin_a);
    if row >= 1 && row <= len && col >= 1 && col <= len
        psf(row, col) = psf(row, col) + 1;
    end
end

total = sum(psf(:));
if total == 0
    psf(center, center) = 1;
else
    psf = psf / total;
end
end
function [u,v] = isoProx(a,b,t)
    % isoProx - Proximal operator of t*||(·,·)||_iso
    %
    % INPUT:
    %   a, b   - paired arrays (same size)
    %   t - threshold parameter
    %
    % OUTPUT:
    %   u, v - proximal result (same size as input)
    
    % Compute magnitude at each position
    r = sqrt(a.^2 + b.^2);
    
    % Compute shrinkage factor: max(1 - lambda/r, 0)
    s = max(1 - t ./ max(r, eps), 0);
    
    % Apply shrinkage to both components
    u = s .* a;
    v = s .* b;
end




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



% test

%% Test 1: Large magnitude (should shrink, not kill)
fprintf('Test 1: Large magnitude...\n');
a1 = 5;
b1 = 0;
lambda1 = 2;
[u1, v1] = isoProx(a1, b1, lambda1);
expected_u1 = 3;
expected_v1 = 0;

fprintf('  Input: a=%.1f, b=%.1f, lambda=%.1f\n', a1, b1, lambda1);
fprintf('  Expected: u=%.1f, v=%.1f\n', expected_u1, expected_v1);
fprintf('  Got:      u=%.1f, v=%.1f\n', u1, v1);
assert(abs(u1 - expected_u1) < 1e-10, 'Test 1 failed!');
assert(abs(v1 - expected_v1) < 1e-10, 'Test 1 failed!');
fprintf('  ✓ Passed\n\n');

%% Test 2: Small magnitude (should kill to zero)
fprintf('Test 2: Small magnitude (killed)...\n');
a2 = 1;
b2 = 1;
lambda2 = 2;
[u2, v2] = isoProx(a2, b2, lambda2);
expected_u2 = 0;
expected_v2 = 0;

fprintf('  Input: a=%.1f, b=%.1f, lambda=%.1f\n', a2, b2, lambda2);
fprintf('  Magnitude: sqrt(1^2 + 1^2) = %.3f < lambda\n', sqrt(2));
fprintf('  Expected: u=%.1f, v=%.1f (killed)\n', expected_u2, expected_v2);
fprintf('  Got:      u=%.1f, v=%.1f\n', u2, v2);
assert(abs(u2 - expected_u2) < 1e-10, 'Test 2 failed!');
assert(abs(v2 - expected_v2) < 1e-10, 'Test 2 failed!');
fprintf('  ✓ Passed\n\n');
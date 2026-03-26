function x_prox = l1Prox(x, t)
% Soft thresholding
% (x_prox)_i = x_i - t if x_i > t
% (x_prox)_i = x_i + t if x_i < -t
% (x_prox)_i = 0 if abs(x_i) <= t

x_prox = sign(x) .* max(abs(x) - t, 0);

end











%% Test 1: Mixed values with small threshold
fprintf('Test 1: Mixed values, small threshold...\n');
x1 = [0.5, -0.3, 0.1, -0.7];
t1 = 0.2;
result1 = l1Prox(x1, t1);
expected1 = [0.3, -0.1, 0, -0.5];
fprintf('  Input:    [%.1f, %.1f, %.1f, %.1f], t=%.1f\n', x1, t1);
fprintf('  Expected: [%.1f, %.1f, %.1f, %.1f]\n', expected1);
fprintf('  Got:      [%.1f, %.1f, %.1f, %.1f]\n', result1);
assert(all(abs(result1 - expected1) < 1e-10), 'Test 1 failed!');
fprintf('  ✓ Passed\n\n');

%% Test 2: Values at and near boundary
fprintf('Test 2: Values at boundary...\n');
x2 = [-0.5, 0.4, 0, 0.8];
t2 = 0.5;

%% Test 3: t = 0 (no shrinkage)
x3 = [1, -2, 3];
t3 = 0;
result3 = l1Prox(x3, t3);
expected3 = [1, -2, 3];  % Should be unchanged
fprintf('Test 3 (t=0): '); disp(result3);
assert(all(abs(result3 - expected3) < 1e-10), 'Test 3 failed!');

%% Test 4: Large threshold (everything becomes zero)
x4 = [0.5, -0.3, 0.8, -0.2];
t4 = 1.0;  % Larger than all |x_i|
result4 = l1Prox(x4, t4);
expected4 = [0, 0, 0, 0];  % All killed
fprintf('Test 4 (large t): '); disp(result4);
assert(all(abs(result4 - expected4) < 1e-10), 'Test 4 failed!');

%% Test 5: Exact boundary values
x5 = [1, -1, 0.5, -0.5];
t5 = 1;
result5 = l1Prox(x5, t5);
expected5 = [0, 0, 0, 0];  % At boundary: |x| = t → 0
fprintf('Test 5 (boundary): '); disp(result5);
assert(all(abs(result5 - expected5) < 1e-10), 'Test 5 failed!');

%% Test 6: Very small threshold
x6 = [5, -3, 2];
t6 = 0.1;
result6 = l1Prox(x6, t6);
expected6 = [4.9, -2.9, 1.9];  % Gentle shrinkage
fprintf('Test 6 (small t): '); disp(result6);
assert(all(abs(result6 - expected6) < 1e-10), 'Test 6 failed!');

%% Test 7: Extreme values
x7 = [100, -100, 0.001];
t7 = 5;
result7 = l1Prox(x7, t7);
expected7 = [95, -95, 0];
fprintf('Test 7 (extreme): '); disp(result7);
assert(all(abs(result7 - expected7) < 1e-10), 'Test 7 failed!');

%% Test 8: Matrix input (not just vectors)
x8 = [2, -1; 0.3, -0.5];
t8 = 0.5;
result8 = l1Prox(x8, t8);
expected8 = [1.5, -0.5; 0, 0];
fprintf('Test 8 (matrix): \n'); disp(result8);
assert(all(abs(result8(:) - expected8(:)) < 1e-10), 'Test 8 failed!');

%% Test 9: All zeros
x9 = [0, 0, 0];
t9 = 1;
result9 = l1Prox(x9, t9);
expected9 = [0, 0, 0];
fprintf('Test 9 (all zeros): '); disp(result9);
assert(all(abs(result9 - expected9) < 1e-10), 'Test 9 failed!');

%% Test 10: Single positive value > t
x10 = 3;
t10 = 1;
result10 = l1Prox(x10, t10);
expected10 = 2;
fprintf('Test 10 (single +): %.1f\n', result10);
assert(abs(result10 - expected10) < 1e-10, 'Test 10 failed!');

%% Test 11: Single negative value < -t
x11 = -3;
t11 = 1;
result11 = l1Prox(x11, t11);
expected11 = -2;
fprintf('Test 11 (single -): %.1f\n', result11);
assert(abs(result11 - expected11) < 1e-10, 'Test 11 failed!');

%% Test 12: Single small value
x12 = 0.5;
t12 = 1;
result12 = l1Prox(x12, t12);
expected12 = 0;
fprintf('Test 12 (single small): %.1f\n', result12);
assert(abs(result12 - expected12) < 1e-10, 'Test 12 failed!');

fprintf('\n✓ All tests passed!\n');
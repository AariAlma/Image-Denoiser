
% l1ShiftProx
function y_prox = l1ShiftProx(y, b, t)
    y_prox = b + l1Prox(y - b, t);
end
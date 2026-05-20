% l2ShiftProx
function y_prox = l2ShiftProx(y, b, t)
    y_prox = b + l2Prox(y - b, t);
end
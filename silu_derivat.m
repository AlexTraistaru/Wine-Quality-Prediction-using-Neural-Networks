%Traistaru Alexandru Mihai
%Grupa 324AA
function dy = silu_derivat(z)
    sigmoid = 1 ./ (1 + exp(-z));
    dy = sigmoid + z .* sigmoid .* (1 - sigmoid); 
end

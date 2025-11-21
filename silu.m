%Traistaru Alexandru Mihai
%Grupa 324AA
function y = silu(z) 
%y = z * sigmoid(z)
    y = z ./ (1 + exp(-z));
end

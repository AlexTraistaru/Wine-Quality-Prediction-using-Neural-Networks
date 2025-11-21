%Traistaru Alexandru Mihai
%Grupa 324AA
function L = pierdere(e_i, y_i)
%Eroare Medie Pătratica (MSE), e_i este o predictie, iar y_i este ce da
%real in urma algoritmului
    N = length(y_i); %lungimea lui y_i
    S = 0;
    for i = 1:N
        S = S + (e_i(i) - y_i(i))^2;
    end
    L = (1/(2*N)) * S;
end

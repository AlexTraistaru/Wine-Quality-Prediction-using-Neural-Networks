%Traistaru Alexandru Mihai
%Grupa 324AA
function [X, x_star, errors, norms, times] = metoda_gradient(A, e, n, m, rata_de_invatare, max_iter, prag_gradient)
    % A - matricea de intrari 
    % e - vectorul de iesiri (etichete reale)
    % n - numarul de intrari fara bias (coloane din A fara bias)
    % m - numarul de neuroni
    % rata_de_invatare - pasul de invatare

    N = size(A, 1); % N - numarul de exemple de antrenare (randuri din A)

    X = randn(n + 1, m); % Initializam aleator marjele (ponderile) pentru conexiunile input - hidden
    x_star = randn(m, 1); % Initializam aleator ponderile pentru conexiunile hidden - iesire

    iter = 0;
    norma_gradient = inf; 
    errors = zeros(1, max_iter); % Vector pentru erori MSE la fiecare iteratie
    norms = zeros(1, max_iter); % Vector pentru norma gradientului la fiecare iteratie
    times = zeros(1, max_iter); % Vector pentru timpul de executie la fiecare iteratie

    while iter < max_iter && norma_gradient > prag_gradient
        iter = iter + 1; 
        tstart = tic; % Pornim timpul
        Z = A * X; % Inmultim intrarile A cu ponderile X pentru a obtine inputurile stratului ascuns
        H = silu(Z); %silu e functia e activare
        y_pred = H * x_star; % Calculam predictia finala
        eroare_vec = y_pred - e; % Vectorul diferentelor dintre predictii si valori reale

        % Calculam gradientul functiei de pierdere fata de parametrii x_star
        dL_dx_star = (1/N) * (H' * eroare_vec); 
        silu_der = silu_derivat(Z);
        % Calculam deltele pentru stratul ascuns
        delta_hidden = eroare_vec * x_star'; 
        % Calculam gradientul pierderii fata de ponderile X
        dL_dX = (1/N) * (A' * (delta_hidden .* silu_der)); 

        % Norma gradientului combina contributiile din ambele seturi de ponderi
        norma_gradient = sqrt(sum(dL_dX(:).^2) + sum(dL_dx_star(:).^2)); 
        % Actualizam ponderile folosind gradientii calculati si rata de invatare
        X = X - rata_de_invatare * dL_dX;
        x_star = x_star - rata_de_invatare * dL_dx_star;

        times(iter) = toc(tstart); % Salvam timpul scurs pentru aceasta iteratie

        % Salvam eroarea medie patrata pentru aceasta iteratie
        errors(iter) = sum(eroare_vec.^2) / (2*N);

        % Salvam norma gradientului pentru aceasta iteratie
        norms(iter) = norma_gradient;

        if mod(iter,100)==0 || iter==1
            fprintf('metoda gradient la iteratia %d, eroarea este %.3f si norma este %.5f\n', iter, errors(iter), norma_gradient);
        end
    end

    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);
end

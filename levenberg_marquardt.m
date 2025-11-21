%Traistaru Alexandru Mihai
%Grupa 324AA
function [X, x_star, errors, norms, times] = metoda_levenberg_marquardt(A, e, n, m, lambda_init, max_iter, prag_gradient)
    % A - matricea de intrari
    % e - vectorul de iesiri (etichete reale)
    % n - numarul de intrari fara bias
    % m - numarul de neuroni
    % lambda_init - valoarea initiala a coeficientului de regularizare

    N = size(A, 1); % N - numarul de exemple de antrenare (randuri din A)

    X = randn(n + 1, m); % Initializare aleatoare a ponderilor intre intrare si strat ascuns
    x_star = randn(m, 1); % Initializare aleatoare a ponderilor strat ascuns - iesire

    iter = 0;
    norma_gradient = inf; % Norma gradientului porneste de la infinit

    errors = zeros(1, max_iter); % Vector pentru erorile de la fiecare iteratie
    norms = zeros(1, max_iter); % Vector pentru normele gradientului
    times = zeros(1, max_iter); % Vector pentru timpii de executie ai fiecarei iteratii

    lambda = lambda_init; % Setam lambda initial (pentru regularizare)

    while iter < max_iter && norma_gradient > prag_gradient
        iter = iter + 1; 
        tstart = tic; % Pornim timpul
        Z = A * X; % Calculam intrarile in stratul ascuns
        H = silu(Z); 
        y_pred = H * x_star; % Calculam predictiile 
        eroare_vec = y_pred - e; % Vectorul erorilor dintre predictii si valori reale
        silu_der = silu_derivat(Z);

        % Jacobianul fata de x_star (iesire strat ascuns)
        Jx = H; % Jx este pur si simplu iesirea stratului ascuns
        % Jacobianul fata de X (intrari strat ascuns)
        JX = zeros(N, (n+1)*m); % Initializam Jacobianul pentru X
        for j = 1:m
            % Pentru fiecare neuron din stratul ascuns
            JX(:, (j-1)*(n+1)+1:j*(n+1)) = A .* (silu_der(:,j) * x_star(j));
            % Extragem derivata fata de ponderile X
        end
        % Jacobianul total
        J_total = [JX, Jx]; 

        gradient = (1/N) * (J_total' * eroare_vec); % Calculam gradientul
        hessiana = (1/N) * (J_total' * J_total); % Hessiana

        hessiana = hessiana + lambda * eye(size(hessiana)); % Adaugam regularizare pentru stabilitate numerica
        d = -hessiana \ gradient; % Calculam directia de coborare
        X_nou = X;
        x_star_nou = x_star;
        index = 1;
        for j = 1:m
            for p = 1:n+1
                % Actualizam X_nou folosind directia d
                X_nou(p,j) = X(p,j) + d(index);
                index = index + 1;
            end
        end

        for j = 1:m
            % Actualizam x_star_nou folosind directia d
            x_star_nou(j) = x_star(j) + d(index);
            index = index + 1;
        end

        Z_nou = A * X_nou; % Calculam eroarea noua
        H_nou = silu(Z_nou);
        y_hat_nou = H_nou * x_star_nou;
        eroare_noua = sum((y_hat_nou - e).^2) / 2; % Eroarea noua (sum patrate erori)

        if eroare_noua < sum((y_pred - e).^2) / 2
            % Daca eroarea noua este mai mica, acceptam pasul
            X = X_nou;
            x_star = x_star_nou;
            lambda = lambda / 10; % Scadem lambda 
        else
            % Daca eroarea nu scade, respingem pasul si crestem lambda,
            % pentru prudenta
            lambda = lambda * 10; 
        end

        norma_gradient = sqrt(sum(gradient.^2)); % Norma gradientului pentru criteriul de oprire

        times(iter) = toc(tstart); % Salvam timpul iteratiei
        errors(iter) = eroare_noua / N; % Salvam eroarea normalizata
        norms(iter) = norma_gradient; % Salvam norma gradientului

        if mod(iter,100)==0 || iter==1
            fprintf('metoda levenberg marquardt la iteratia %d, eroarea este de %.3f, norma este %.5f\n', iter, errors(iter), norma_gradient);
        end
    end

    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);
end
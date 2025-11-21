%Traistaru Alexandru Mihai
%Grupa 324AA
function [X, x_star, erori, norme, times] = metoda_newton(A, e, n, m, max_iter, prag_gradient)
    % A - matricea de intrari (bias inclus)
    % e - vectorul de iesiri (etichete reale)
    % n - numarul de intrari fara bias
    % m - numarul de neuroni

    N = size(A, 1); % N - numar de exemple de antrenare

    X = randn(n + 1, m); % Initializam aleator ponderile intre input si stratul ascuns
    x_star = randn(m, 1); % Initializam aleator ponderile intre stratul ascuns si iesire

    iter = 0; 
    norma_gradient = inf; 

    erori = zeros(1, max_iter); % Vector pentru erorile MSE la fiecare iteratie
    norme = zeros(1, max_iter); % Vector pentru normele gradientului
    times = zeros(1, max_iter); % Vector pentru timpii fiecarei iteratii

    while iter < max_iter && norma_gradient > prag_gradient
        iter = iter + 1; 
        tstart = tic; % Pornesc timpul

        Z = A * X; % Calculam inputurile in stratul ascuns
        H = silu(Z); 
        y_pred = H * x_star; % Calculam predictiile

        eroare_vec = y_pred - e; % Diferenta intre predictii si iesiri reale

        dL_dx_star = (1/N) * (H' * eroare_vec); % Gradient fata de x_star
        silu_der = silu_derivat(Z);

        delta_hidden = eroare_vec * x_star'; 
        dL_dX = (1/N) * (A' * (delta_hidden .* silu_der)); % Gradient fata de X

        % Combinam gradientii
        grad = [dL_dX(:); dL_dx_star];

        dim = (n+1)*m + m; % Dimensiunea totala (X + x_star)
        J = zeros(N, dim); % Jacobianul

        for j = 1:m
            % Completam Jacobianul pentru fiecare neuron 
            J(:, (j-1)*(n+1)+1:j*(n+1)) = A .* (silu_der(:,j) * x_star(j));
        end
        % Completam Jacobianul pentru iesire
        J(:, (n+1)*m+1:end) = H;

        % Hessianul complet aproximat
        H_ = (1/N) * (J' * J);

        aux = 1e-3; % Termen de regularizare pentru stabilitate numerica
        d = -(H_ + aux * eye(size(H_,1))) \ grad; % Directia de coborare Newton regularizata

        index = 1;
        for j = 1:m
            for p = 1:n+1
                X(p,j) = X(p,j) + d(index); % Actualizam X
                index = index + 1;
            end
        end
        for j = 1:m
            x_star(j) = x_star(j) + d(index); % Actualizam x_star
            index = index + 1;
        end

        norma_gradient = sqrt(sum(grad.^2)); % Norma gradientului pentru criteriul de oprire

        times(iter) = toc(tstart); % Timpul scurs pentru iteratia curenta
        erori(iter) = sum((y_pred - e).^2) / (2*N); % Eroarea medie patrata
        norme(iter) = norma_gradient; % Norma gradientului

        if mod(iter,100)==0 || iter==1
            fprintf('metoda newton la iteratia %d, eroarea este de %.3f si norma este %.5f\n', iter, erori(iter), norma_gradient);
        end
    end

    erori = erori(1:iter);
    norme = norme(1:iter);
    times = times(1:iter);
end

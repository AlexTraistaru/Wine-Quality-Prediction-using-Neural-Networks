%Traistaru Alexandru Mihai
%Grupa 324AA
clc; 
clear;

baza_de_date = readmatrix('winequality-white.csv', 'Delimiter', ';'); 
%citesc fisierul csv iar delimitatorul este ";" pentru ca fisierul are separator ";"
baza_fara_etichete = baza_de_date(:, 1:end-1);
%toate coloanele, mai putin ultima, acolo sunt etichetele
etichete = baza_de_date(:, end);
%aici este scorul vinurilor, etichetele pentru regresie


% Normalizare pentru fiecare coloana, face media 0 si deviatia standard 1
% Ajuta la stabilitatea algoritmului
medie = mean(baza_fara_etichete);
%mean calculeaza media pe coloane
dev_standard = std(baza_fara_etichete);
%std calculeaza deviatia standard pe coloane
x_norm = (baza_fara_etichete - medie) ./ dev_standard;
%toate feature urile (coloanele) au aceeasi scara, ceea ce face invatarea mai rapida si
%stabila si evita ca o coloana cu valori mari sa le domine pe celelalte


% Adaugare bias. Bias este o coloana de 1 la final, pentru a deplasa
% functia de activare, nu trebuie sa treaca prin origine. De exemplu, daca
% intrarile sunt mici (aproape de 0) si functia trece prin (0,0), iesirea
% poate fi slaba
bias = [x_norm, ones(size(baza_fara_etichete,1),1)]; %ones construieste o coloana de 1


% impart baza de date in 80% pentru antrenare si 20% pentru testare
N = size(bias, 1); %nr de randuri din bias, adica cate instante am, cate vinuri
baza_amestec = randperm(N); %randperm amesteca datele in cazul in care am doar vinuri bune la inceput de exemplu
prag_baza = round(0.8 * N); %round rotunjeste la cel mai apropiat intreg, pentru ca numarul de raduri in x_antrenament sa fie intreg
x_antrenament = bias(baza_amestec(1:prag_baza), :); %intrari pentru antrenare
y_antrenament = etichete(baza_amestec(1:prag_baza)); %etichete pentru antrenare
x_test  = bias(baza_amestec(prag_baza+1:end), :); %intrari pentru testare
y_test  = etichete(baza_amestec(prag_baza+1:end)); %etichete pentru testare

[n, ~] = size(x_antrenament');  %nr de intrari n = 12 (11 coloane + 1 bias), am pus "~" ca sa ignor nr de coloane pt ca size imi intorcea si nr de randuri si nr de coloane
%transpusa pt ca matlab reprezinta matricea pe coloane
m = 10; %neuroni din strat ascuns
max_iter = 1000;
prag = 1e-6;

fprintf('\nmetoda gradient\n');
[Xg, xg, err_g, grad_g, time_g] = metoda_gradient(x_antrenament, y_antrenament, n-1, m, 0.01, max_iter, prag);

fprintf('\nmetoda levenberg marquardt\n');
[Xlm, xlm, err_lm, grad_lm, time_lm] = levenberg_marquardt(x_antrenament, y_antrenament, n-1, m, 0.01, max_iter, prag);

fprintf('\nmetoda newton\n');
[Xn, xn, err_n, grad_n, time_n] = metoda_newton(x_antrenament, y_antrenament, n-1, m, max_iter, prag);

% R² pentru toate metodele
yp_g  = silu(x_test * Xg) * xg;
yp_lm = silu(x_test * Xlm) * xlm;
yp_n = silu(x_test * Xn) * xn;

R2_g  = 1 - sum((y_test - yp_g).^2)  / sum((y_test - mean(y_test)).^2);
R2_lm = 1 - sum((y_test - yp_lm).^2) / sum((y_test - mean(y_test)).^2);
R2_gn = 1 - sum((y_test - yp_n).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\nScor R² Gradient: %.3f\n', R2_g);
fprintf('Scor R² Levenberg–Marquardt: %.3f\n', R2_lm);
fprintf('Scor R² Newton: %.3f\n', R2_gn);

% MSE pentru toate metodele
MSE_g  = mean((y_test - yp_g).^2);
MSE_lm = mean((y_test - yp_lm).^2);
MSE_n  = mean((y_test - yp_n).^2);

fprintf('\nMSE Gradient: %.3f\n', MSE_g);
fprintf('MSE Levenberg–Marquardt: %.3f\n', MSE_lm);
fprintf('MSE Newton: %.3f\n', MSE_n);

% metoda gradient
figure('Name','Metoda Gradient');

subplot(4,1,1);
semilogy(err_g, 'b', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Eroare');
title('Eroarea dupa iteratii');
grid on;

subplot(4,1,2);
semilogy(grad_g, 'r', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Norma Gradient');
title('Norma dupa iteratii');
grid on;

subplot(4,1,3);
semilogy(cumsum(time_g), err_g, 'b', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Eroare');
title('Eroarea dupa timp');
grid on;

subplot(4,1,4);
semilogy(cumsum(time_g), grad_g, 'r', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Norma Gradient');
title('Norma dupa timp');
grid on;

% metoda levenberg marquardt
figure('Name','Metoda Levenberg-Marquardt');

subplot(4,1,1);
semilogy(err_lm, 'm', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Eroare');
title('Eroarea dupa iteratii');
grid on;

subplot(4,1,2);
semilogy(grad_lm, 'k', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Norma Gradient');
title('Norma dupa iteratii');
grid on;

subplot(4,1,3);
semilogy(cumsum(time_lm), err_lm, 'm', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Eroare');
title('Eroarea dupa timp');
grid on;

subplot(4,1,4);
semilogy(cumsum(time_lm), grad_lm, 'k', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Norma Gradient');
title('Norma dupa timp');
grid on;

% metoda newton
figure('Name','Metoda Newton');

subplot(4,1,1);
semilogy(err_n, 'g', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Eroare');
title('Eroarea dupa iteratii');
grid on;

subplot(4,1,2);
semilogy(grad_n, 'c', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Norma Gradient');
title('Norma dupa iteratii');
grid on;

subplot(4,1,3);
semilogy(cumsum(time_n), err_n, 'g', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Eroare');
title('Eroarea dupa timp');
grid on;

subplot(4,1,4);
semilogy(cumsum(time_n), grad_n, 'c', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Norma Gradient');
title('Norma dupa timp');
grid on;

figure('Name','Comparatie - Norma si Eroare');

% Norma Gradientului in functie de iteratii
subplot(2,2,1);
semilogy(grad_g, 'b', 'LineWidth', 1.5); hold on;
semilogy(grad_lm, 'm', 'LineWidth', 1.5);
semilogy(grad_n, 'g', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Norma Gradient');
title('Norma Gradient vs Iteratii');
legend('Gradient', 'Levenberg–Marquardt', 'Newton');
grid on;

% Norma Gradientului in functie de timp
subplot(2,2,2);
semilogy(cumsum(time_g), grad_g, 'b', 'LineWidth', 1.5); hold on;
semilogy(cumsum(time_lm), grad_lm, 'm', 'LineWidth', 1.5);
semilogy(cumsum(time_n), grad_n, 'g', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Norma Gradient');
title('Norma Gradient vs Timp');
legend('Gradient', 'Levenberg–Marquardt', 'Newton');
grid on;

% Eroarea in functie de iteratii
subplot(2,2,3);
semilogy(err_g, 'b', 'LineWidth', 1.5); hold on;
semilogy(err_lm, 'm', 'LineWidth', 1.5);
semilogy(err_n, 'g', 'LineWidth', 1.5);
xlabel('Iteratii'); ylabel('Eroare (MSE)');
title('Eroare vs Iteratii');
legend('Gradient', 'Levenberg–Marquardt', 'Newton');
grid on;

% Eroarea in functie de timp
subplot(2,2,4);
semilogy(cumsum(time_g), err_g, 'b', 'LineWidth', 1.5); hold on;
semilogy(cumsum(time_lm), err_lm, 'm', 'LineWidth', 1.5);
semilogy(cumsum(time_n), err_n, 'g', 'LineWidth', 1.5);
xlabel('Timp (s)'); ylabel('Eroare (MSE)');
title('Eroare vs Timp');
legend('Gradient', 'Levenberg–Marquardt', 'Newton');
grid on;

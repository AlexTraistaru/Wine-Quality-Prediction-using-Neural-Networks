# Wine Quality Prediction using Neural Networks – Optimization Methods Comparison

This project implements and trains a **single-layer neural network** to predict
white wine quality based on 11 physico-chemical features.  
Three numerical optimization algorithms are compared during the training
process: **Gradient Descent**, **Levenberg–Marquardt**, and **Newton’s Method**.

The main objective is to analyze how different optimization strategies affect
convergence speed, Mean Squared Error, gradient norm, and prediction accuracy.

---

## 1. Problem Overview

We address a supervised **regression** task:

\[
x \in \mathbb{R}^{11} \longrightarrow \hat{y} = f(Wx) 
\]

where:

- \( f \) is the linear activation function of the neural network
- \( W \) represents the trainable weights
- The model minimizes **Mean Squared Error (MSE)**.

Although simple, this architecture qualifies as a **neural network** because
it consists of an input layer, a weighted linear transformation, and a single
output neuron.

---

## 2. Dataset

The project uses the `winequality-white.csv` dataset with:

- **4898 samples**
- **11 input features** (acidity, residual sugar, density, pH, alcohol, etc.)
- **1 regression label** (`quality`, integer between 3 and 9)

All input features are **standardized** before training.

---

## 3. Neural Network Architecture

- **Type:** Single-layer neural network (linear regression model)
- **Input dimension:** 11 features  
- **Output dimension:** 1  
- **Activation:** Identity (linear output)
- **Trainable parameters:** Weight vector \( W \) + bias term

Even though the activation is linear, the model is trained using methods
common in neural networks, such as Levenberg–Marquardt and second-order
optimization.

---

## 4. Optimization Methods

Three numerical methods are implemented and compared:

### 4.1 Gradient Descent
- First-order method
- Constant or tuned learning rate
- Slow but stable convergence

### 4.2 Levenberg–Marquardt (LM)
- Hybrid between Gradient Descent and Newton
- Adds a damping coefficient λ to stabilize Hessian inversion
- Often the **best performer** for neural network training in small-scale problems

### 4.3 Newton’s Method
- Second-order optimization using the analytical Hessian
- Very fast theoretical convergence
- Requires Hessian regularization for numerical stability

---

## 5. Implementation Structure

Typical MATLAB file organization:

- `main.m` – loads dataset, initializes weights, runs all methods
- `gradient_method.m` – Gradient Descent implementation
- `levenberg_marquardt.m` – LM implementation
- `newton_method.m` – Newton implementation
- `winequality-white.csv` – dataset
- `plots/` – optional folder for generated graphs

All scripts compute and plot:

- MSE vs iterations  
- MSE vs time  
- Gradient norm vs iterations  
- Gradient norm vs time  

and display intermediate training logs (iteration 1, 100, ..., 1000).

---

## 6. Training Process

1. Data is normalized  
2. Network weights \( W \) are initialized  
3. Each optimization algorithm updates the weights:  
   - Compute gradient  
   - Compute or approximate Hessian  
   - Apply update rule  
4. Performance metrics recorded:
   - Mean Squared Error  
   - Gradient norm  
   - Runtime  
   - R² score  

---

## 7. Results Summary

- **Gradient Descent**  
  - Slowest convergence  
  - Stable decrease in error  

- **Levenberg–Marquardt**  
  - Best trade-off between stability and speed  
  - Achieves lowest MSE and highest R²  

- **Newton’s Method**  
  - Fast reduction of gradient norm  
  - Sensitive to Hessian conditioning  
  - Requires regularization  

Overall, **Levenberg–Marquardt** performs best for this neural network training task.

# Spectral Interpolation Classifier

A non-iterative classification approach that combines distance-weighted interpolation with Fourier-based functional approximation to model data without gradient-based training.

---

## 🚀 Overview

This project explores an alternative to conventional machine learning pipelines by replacing iterative parameter learning with **closed-form functional reconstruction**.

The method:

* Uses **distance-weighted KNN interpolation** to capture local structure
* Projects high-dimensional data into a **1D locality-preserving representation**
* Approximates the resulting function using a **truncated Fourier series**
* Enables prediction using **precomputed spectral coefficients**, avoiding iterative training

---

## 🧠 Key Idea

Instead of learning model parameters through optimization (e.g., gradient descent), this approach:

> **Reconstructs the input-output relationship directly from data using interpolation + spectral approximation**

This results in:

* Reduced training complexity
* Compact representation of training data
* Competitive performance on structured datasets

---

## ⚙️ Method Pipeline

1. **Input Normalization**
   Scale and preprocess input features

2. **Local Interpolation**
   Apply distance-weighted KNN to estimate output values

3. **Dimensional Projection**
   Map high-dimensional inputs to a 1D space while preserving locality

4. **Fourier Approximation**
   Fit a truncated Fourier series to the interpolated function

5. **Prediction**
   Use spectral coefficients to compute outputs for new inputs

---

## 📊 Results

| Dataset | Accuracy |
| ------- | -------- |
| MNIST   | ~95%     |
| Iris    | ~80%     |

* Performance improves with higher Fourier approximation order
* Trade-off observed between accuracy and computational cost

---

## 📈 Characteristics

* **Non-iterative training** (closed-form coefficient computation)
* **Interpolation-driven modeling**
* **Spectral compression of training data**
* Suitable for problems where **spatial locality encodes class structure**

---

## ⚠️ Limitations

* Performance depends on quality of **dimensional projection**
* Fourier approximation assumes a degree of **function smoothness**
* May struggle with highly irregular decision boundaries

---

## 📂 Repository Structure

```
/notebooks  → experiments and visualization  
/report     → project PDF  
```

---

## 📄 Report

Full technical report available here:
[SLP Report (PDF)](./report/SLP_Report.pdf)

---

## 🔧 Future Work

* Formal analysis of locality-preserving projection
* Comparison with kernel methods and neural networks
* Optimization of projection and approximation stages
* Extension to higher-dimensional spectral representations

---

## 👤 Author

Samarth Pusalkar
IIT Bombay – Aerospace Engineering
Guide - Prof. Harshad Khadilkar
---

## 📌 Note

This project was developed as part of a supervised learning project under academic guidance and is intended as an exploration of alternative modeling techniques.

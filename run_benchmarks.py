import os
import gzip
import time
import numpy as np
import scipy.integrate as spi
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.model_selection import train_test_split
import sklearn.datasets
import warnings

# Suppress integration warnings
warnings.filterwarnings("ignore", category=spi.IntegrationWarning)

# --- 1. Load Functions from the Notebook ---

def mapd(x, y=1):
    D = x.shape[1]
    z_powers = (0.01j) ** np.arange(D - 1, -1, -1)
    m1_arr = np.dot(x, z_powers)
    return y * (m1_arr - 1j * m1_arr) + x.sum(axis=1)

def conv(x_train):
    # Standardizing calculation (from notebook)
    xw = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0) + 1e-9)
    return mapd(x_train)

def sortItOut(xx, Y_train):
    xc = sorted(xx)
    inid = list(range(len(xc)))
    inid.sort(key=lambda idx: xx[idx])
    yc = Y_train[inid]
    xc = np.array(xc)
    yc = np.array(yc)
    return xc, yc

def uM(xc, yc):
    uniqueDict = {}
    for i in range(len(xc)):
        uniqueDict[xc[i]] = []
    for i in range(len(xc)):
        uniqueDict[xc[i]].append(yc[i])
    for i in uniqueDict:
        uniqueDict[i] = np.array(uniqueDict[i]).mean()
    yc_ = []
    for i in xc:
        yc_.append(uniqueDict[i])
    return xc, np.array(yc_)

def KNN_Classify_(x_train, Y_train, inpus, P_val):
    inpus = np.array(inpus).reshape(-1, 1) % (round(P_val) + 1e-9)
    Ans = []
    n = len(x_train)
    
    # Subsampling as in notebook
    idx = np.random.randint(0, x_train.shape[0], min(7000, n))
    X_train_sub = x_train[idx]
    y_train_sub = Y_train[idx]
    
    tree = KDTree(X_train_sub)
    k_val = min(700, max(1, int(n / 3)))
    distances, indices = tree.query(inpus, k=k_val)
    
    for ii in range(len(inpus)):  
        dis = distances[ii]
        bias = y_train_sub.mean(axis=0)
        bb = bias.mean()
        k = 2 * np.pi
        bias -= bb
        DD = ((1 / (np.e**(k * dis) + 1).reshape(-1, 1))).sum()
        if DD == 0:
            DD = 1e-9
        term = (((y_train_sub[indices[ii]]) / ((np.e**(k * dis) + 1).reshape(-1, 1))).sum(axis=0) / DD) - bias
        Ans.append(term)
    return Ans

def giveInterpolFunc(xc, yc):
    XX = xc.real
    f = lambda t: KNN_Classify_(XX.reshape(-1, 1), np.array(yc).reshape(-1, 1), [[t]], XX[-1])
    return f

def compute_real_fourier_coeffs(func, N, P_val):
    result = []
    for n in range(N+1):
        an = (2. / P_val) * spi.quad(lambda t: float(func(t)[0][0]) * np.cos(2 * np.pi * n * t / P_val), 0, P_val, limit=50)[0]
        bn = (2. / P_val) * spi.quad(lambda t: float(func(t)[0][0]) * np.sin(2 * np.pi * n * t / P_val), 0, P_val, limit=50)[0]
        result.append((an, bn))
    return np.array(result)

def fit_func_by_fourier_series_with_real_coeffs(t, AB, P_val=25.0):
    result = 0.
    A = AB[:, 0]
    B = AB[:, 1]
    for n in range(0, len(AB)):
        if n > 0:
            result += A[n] * np.cos(2. * np.pi * n * t / P_val) + B[n] * np.sin(2. * np.pi * n * t / P_val)
        else:
            result += A[0] / 2.
    return result

def predd(x_test, AB, P_val=25.0):
    xx = conv(x_test).real
    return fit_func_by_fourier_series_with_real_coeffs(xx, AB, P_val)

def reduceSupervised(x_train, Y_train, N=8):
    xx = conv(x_train)
    xc, yc = sortItOut(xx, Y_train)
    xc, yc = uM(xc, yc)
    f = giveInterpolFunc(xc, yc)
    P_val = max(1.0, round(xc[-1].real))
    AB = compute_real_fourier_coeffs(f, N, P_val)
    return AB, P_val


# --- 2. Dataset Loading Helpers ---

def load_mnist(data_dir="./data/mnist", subset="train"):
    prefix = "train" if subset == "train" else "t10k"
    img_path = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte.gz")
    lbl_path = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte.gz")
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    with gzip.open(lbl_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    labels = labels.astype(np.int64)
    return images, labels


# --- 3. Run Experiments ---

print("=" * 60)
print("RUNNING PROFILING AND BENCHMARKING EXPERIMENTS")
print("=" * 60)

# Experiment 1: The Latency Micro-Benchmark (Inference Speedup)
print("\n--- EXPERIMENT 1: Latency Micro-Benchmark (Inference Speedup) ---")
print("Loading MNIST dataset...")
x_train, y_train = load_mnist(subset="train")
x_test, y_test = load_mnist(subset="test")

# Standard KNN Benchmarking
print("Training standard scikit-learn KNN on full 60k MNIST training samples...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

print("Timing standard KNN prediction on 1,000 test samples...")
test_subset = x_test[:1000]
t0 = time.perf_counter()
knn_preds = knn.predict(test_subset)
knn_time = time.perf_counter() - t0
print(f"Standard KNN Inference Latency: {knn_time:.4f} seconds")

# Fourier Classifier Benchmarking
print("Training Spectral Interpolation Classifier on subset to obtain AB coefficients...")
# Using 500 samples for train to avoid slow numerical integration
x_train_sub = x_train[:500]
y_train_sub = y_train[:500]
AB, P_val = reduceSupervised(x_train_sub, y_train_sub, N=25)
print(f"Spectral coefficients shape: {AB.shape}, Period P: {P_val}")

print("Timing Spectral Interpolation Classifier predd() on 1,000 test samples...")
t0 = time.perf_counter()
fourier_preds = predd(test_subset, AB, P_val)
fourier_time = time.perf_counter() - t0
print(f"Spectral Classifier Inference Latency: {fourier_time:.4f} seconds")

speedup = knn_time / fourier_time
print(f"Inference Speedup: {speedup:.2f}x")


# Experiment 2: The Storage & Memory Footprint (Compression Ratio)
print("\n--- EXPERIMENT 2: Storage & Memory Footprint (Compression Ratio) ---")
# MNIST training array is 60000 x 784 float32
raw_memory = 60000 * 784 * 4 # bytes
# AB is 25x2 complex/float32 (here it is 26 x 2 floats)
ab_memory = AB.nbytes # bytes

raw_mb = raw_memory / (1024 * 1024)
ab_bytes = ab_memory

compression_ratio = (1.0 - (ab_memory / raw_memory)) * 100.0
print(f"Raw MNIST Training Matrix Memory Footprint: {raw_mb:.2f} MB")
print(f"Spectral Coefficients Matrix Memory Footprint: {ab_bytes} bytes")
print(f"Data Compression Ratio: {compression_ratio:.6f}%")


# Experiment 3: The Pareto Frontier Sweep (Accuracy vs. Parameters)
print("\n--- EXPERIMENT 3: The Pareto Frontier Sweep (Accuracy vs. Parameters) ---")
print("Loading Iris dataset...")
X, y = sklearn.datasets.load_iris(return_X_y=True)
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(X, y, test_size=0.3, random_state=42)

N_values = [5, 10, 25, 50, 100]
print(f"{'N':<5} | {'Train Time (s)':<15} | {'Test Accuracy (%)':<20}")
print("-" * 50)

for N in N_values:
    t0 = time.perf_counter()
    AB_iris, P_iris = reduceSupervised(x_train_iris, y_train_iris, N=N)
    train_time = time.perf_counter() - t0
    
    y_pred_iris = np.array(list(map(round, predd(x_test_iris, AB_iris, P_iris))))
    acc = np.mean(y_pred_iris == y_test_iris) * 100.0
    
    print(f"{N:<5} | {train_time:<15.4f} | {acc:<20.2f}%")


# Experiment 4: The Projection Bottleneck Profile
print("\n--- EXPERIMENT 4: The Projection Bottleneck Profile ---")
# Profile test phase on MNIST subset
t0 = time.perf_counter()
xx = conv(test_subset).real
projection_time = time.perf_counter() - t0

t0 = time.perf_counter()
fit_func_by_fourier_series_with_real_coeffs(xx, AB, P_val)
fourier_eval_time = time.perf_counter() - t0

total_eval_time = projection_time + fourier_eval_time
proj_percentage = (projection_time / total_eval_time) * 100.0
fourier_percentage = (fourier_eval_time / total_eval_time) * 100.0

print(f"Projection (mapd/conv) Latency: {projection_time:.6f} seconds ({proj_percentage:.2f}%)")
print(f"Fourier Evaluation Latency: {fourier_eval_time:.6f} seconds ({fourier_percentage:.2f}%)")
print(f"Total Prediction Pipeline Latency: {total_eval_time:.6f} seconds")
print("=" * 60)

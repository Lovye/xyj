# 快速傅里叶变换（FFT）

快速傅里叶变换（FFT, Fast Fourier Transform）是一种高效计算离散傅里叶变换（DFT）的算法。DFT 将时间域信号转换为频率域信号，FFT 则通过减少计算复杂度使其更高效。

---

## 目录
1. 什么是傅里叶变换？
2. 离散傅里叶变换（DFT）
3. 快速傅里叶变换（FFT）的原理
4. FFT 的实现步骤
5. 应用场景
6. Python 中的 FFT 示例

---

## 什么是傅里叶变换？

傅里叶变换是将信号从 **时间域**（或空间域）转换为 **频率域** 的工具。其基本思想是将信号分解为不同频率的正弦波组合。

**公式：**

连续傅里叶变换的公式为：
$F(k) = \int_{-\infty}^{\infty} f(t) e^{-j 2 \pi k t} \, dt$

其中：
- $f(t)$：时间域信号
- $F(k)$：频率域信号
- $e^{-j 2 \pi k t}$：复指数函数

---

## 离散傅里叶变换（DFT）

由于计算机无法处理连续信号，傅里叶变换通常离散化，即通过 DFT 进行计算。

**公式：**

$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}, \quad k = 0, 1, \dots, N-1$

其中：
- $x[n]$：输入信号的第 $n$ 个采样点
- $X[k]$：频率域第 $k$ 个采样点
- $N$：信号长度

### DFT 的计算复杂度

- DFT 的直接计算复杂度为 $O(N^2)$，因为每个点需要 $N$ 次运算，总共 $N$ 个点。

---

## 快速傅里叶变换（FFT）的原理

FFT 是一种高效计算 DFT 的算法，基于以下两点：
1. **分而治之**：将一个 $N$-点序列分解为两个 $\frac{N}{2}$-点序列。
2. **利用对称性**：傅里叶变换的系数 $e^{-j \frac{2\pi}{N}}$ 的周期性和对称性。

### FFT 的计算复杂度

通过分而治之，FFT 将计算复杂度降低到 $O(N \log N)$，大幅提升效率。

---

## FFT 的实现步骤

1. **输入信号长度为 $N$ 的序列**：
   - 若 $N$ 不是 2^k 的形式，可用零填充到最近的 2^k 长度。

2. **分解信号**：
   - 将信号分解为偶数索引部分 $x_{even}[n]$ 和奇数索引部分 $x_{odd}[n]$。

3. **递归计算 DFT**：
   - 分别对 $x_{even}$ 和 $x_{odd}$ 递归计算 DFT。

4. **合并结果**：
   - 利用以下公式合并：
     $$
     X[k] = X_{even}[k] + W_N^k \cdot X_{odd}[k]
     $$
     $$
     X[k+N/2] = X_{even}[k] - W_N^k \cdot X_{odd}[k]
     $$
     其中 $W_N^k = e^{-j \frac{2\pi}{N} k}$。

---

## 应用场景

1. **信号处理**：
   - 音频、视频信号分析和滤波。
2. **图像处理**：
   - 压缩、去噪和边缘检测。
3. **频谱分析**：
   - 分析信号的频率分量。
4. **通信**：
   - 调制解调、OFDM 和频谱共享。

---

## Python 中的 FFT 示例

使用 Python 中的 `numpy` 库，我们可以轻松实现 FFT。

### 示例代码
```python
import numpy as np
import matplotlib.pyplot as plt

# 生成时间域信号
fs = 500  # 采样率 (Hz)
t = np.arange(0, 1, 1/fs)  # 时间向量
f1, f2 = 50, 120  # 两个信号的频率 (Hz)
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# FFT
N = len(signal)
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, 1/fs)

# 取绝对值并截取前一半频率
amplitude = np.abs(fft_result[:N//2])
frequencies = frequencies[:N//2]

# 绘制频谱
plt.figure(figsize=(10, 5))
plt.plot(frequencies, amplitude)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
```
**推荐学习资源**： 
* 《信号与系统》 - 理解傅里叶变换理论基础。 
* [Scipy 文档](https://scipy.org/) - Python 科学计算工具。 
* [FFT 可视化工具](https://academo.org/demos/fourier-transform-sine-wave/) - 在线互动 FFT 可视化工具
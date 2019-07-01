import soundfile as sf
import matplotlib.pyplot as plt 
import numpy as np 

s1, s_r = sf.read("s1.wav")
s2, s_r = sf.read("s2.wav")
s1_hat, s_r = sf.read("s1_hat.wav")
s2_hat, s_r = sf.read("s2_hat.wav")

len_min = max(len(s1), len(s2), len(s1_hat), len(s2_hat))
x1 = np.linspace(0, len(s1)/s_r, len(s1))
x1_hat = np.linspace(0, len(s1_hat)/s_r, len(s1_hat))
x2 = np.linspace(0, len(s2)/s_r, len(s2))
x2_hat = np.linspace(0, len(s2_hat)/s_r, len(s2_hat))

"""
fig = plt.figure(figsize=(10, 9))

ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412, sharey=ax1)
ax3 = fig.add_subplot(413, sharey=ax1)
ax4 = fig.add_subplot(414, sharey=ax1)

ax1.plot(x1, s1)
ax1.set_xlim(0, 4.5)
ax1.set_title("s1")
ax1.set_xlabel("time [s]")
ax2.plot(x1_hat, s1_hat)
ax2.set_title("s1_hat")
ax2.set_xlim(0, 4.5)
ax2.set_xlabel("time [s]")
ax3.plot(x2, s2)
ax3.set_xlim(0, 4.5)
ax3.set_title("s2")
ax3.set_xlabel("time [s]")
ax4.plot(x2_hat, s2_hat)
ax4.set_xlim(0, 4.5)
ax4.set_title("s2_hat")
ax4.set_xlabel("time [s]")
fig.tight_layout()
plt.savefig("ssss.png")
"""

"""
ss1hat = np.correlate(s1, s1_hat, 'full').max() / len(s1_hat)
ss1 = np.sqrt(np.mean(s1*s1)*np.mean(s1_hat*s1_hat))
print("ss1-ss1hat = ", ss1hat / ss1)

ss2hat = np.correlate(s2, s2_hat, 'full')
plt.figure()
plt.plot(ss2hat)
plt.show()
ss2hat = ss2hat.max() / len(s2_hat)
ss2 = np.sqrt(np.mean(s2*s2)*np.mean(s2_hat*s2_hat))
print("ss2-ss1hat = ", ss2hat / ss2)
"""

def ncorr(a, v):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    v = (v - np.mean(v)) / np.std(v)
    return np.correlate(a, v, 'full')

ss1 = ncorr(s1, s1_hat).max()
ss2 = ncorr(s2, s2_hat).max()
print("ss1:", ss1)
print("ss2:", ss2)

x5, _ = sf.read("x5.wav")
sx = ncorr(s1, x5).max()
print("sx", sx)
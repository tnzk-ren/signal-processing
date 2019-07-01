import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt 


def tau(theta, d):
    c = 340 # m / s
    return d * np.sin(theta) / c

def Lms(inputs, desire, len_adf, mu):
    len_x = len(inputs[0])
    adf = np.zeros((len(inputs), len_adf))
    s = np.zeros(len_x)
    print(len(inputs))
    for i in range(len_adf, len_x):
        x = inputs[:, i-len_adf:i]
        y_sum = np.sum(x * adf)
        error = desire[i] - y_sum
        s[i-len_adf] = error
        gvec = mu * error * x
        adf = adf + gvec
    
    print(s.shape)
    return s, adf

    
def Griffith_jim(x, theta, d, adf_step=129, stepsize=0.05):
    plt.figure()
    plt.plot(x[0])
    x_t = []
    sampling_rate = 16000
    for i in range(len(x)):
        x_i = x[i]
        delay = int(tau(theta, d[i]) * sampling_rate)
        #print(delay)
        zero = np.zeros(delay)
        x_i = np.hstack([zero, x_i])
        x_t.append(x_i)

    lengs = [len(i) for i in x_t]
    len_max = max(lengs)

    for i in range(len(x)):
        len_x = len(x_t[i])
        len_zero = len_max - len_x
        if len_zero > 0:
            zero = np.zeros(len_zero)
            x_t[i] = np.hstack([x_t[i], zero])          
    
    s = np.sum(x_t, axis=0)
    s = s / 5 
    delay_adf = np.zeros((adf_step-1))
    tail_adf = np.zeros(adf_step-1 - len(delay_adf))
    s = np.hstack([delay_adf, s, tail_adf])

    x_s = []
    delay_adf = np.zeros(adf_step-1)
    for i in range(len(x)):
        x_t[i] = np.hstack([delay_adf, x_t[i]])
    
    for i in range(len(x)-1):
        x1 = x_t[i]
        x2 = x_t[i+1]
        x_s.append(x1 - x2)
    x_s = np.array(x_s)
    s_hat, adf = Lms(x_s, s, adf_step, stepsize)

    return s_hat

if __name__ == '__main__':
    n_array = 5
    x = []
    for i in range(n_array):
        data, sampling_rate = sf.read("x{}.wav".format(i+1))
        x.append(data)
    d = 5 / 100
    theta = 0
    theta = 2 * np.pi * theta / 360
    ds1 = [2*d, 0, d, d, 0]
    xs1 = [x[2], x[0], x[1], x[3], x[4]]
    s = Griffith_jim(xs1, theta, ds1)
    sf.write("s1_hat.wav", s, sampling_rate)

    theta = 60
    theta = 2 * np.pi * theta / 360
    ds2 = [4*d, 3*d, 2*d, d, 0]
    xs2 = [x[4], x[3], x[2], x[1], x[0]]
    s = Griffith_jim(xs2, theta, ds2)
    sf.write("s2_hat.wav", s, sampling_rate)
        

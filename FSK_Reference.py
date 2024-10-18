import numpy as np
import pylab as pl
import scipy.signal as signal
from numpy.random import sample

# the following variables setup the system
F_carrier = 1000       # simulate a carrier frequency of 1kHz
F_bit = 50       # simulated bitrate of data
F_dev = 500      # frequency deviation, make higher than bitrate
N = 16          # how many bits to send
A = 1           # transmitted signal amplitude
F_sampling = 10000      # sampling frequency for the simulator, must be higher than twice the carrier frequency
A_n = 1     # noise peak amplitude
N_prntbits = 10 # number of bits to print in plots

def plot_data(y):
    # view the data in time and frequency domain
    # calculate the frequency domain for viewing purposes
    N_FFT = float(len(y))
    f = np.arange(0, F_sampling/2, F_sampling/N_FFT)
    w = np.hanning(len(y))
    y_f = np.fft.fft(np.multiply(y, w))
    y_f = 10 * np.log10(np.abs(y_f[0:int(N_FFT//2)]/N_FFT))
    
    pl.subplot(3,1,1)
    pl.plot(t[0:int(F_sampling*N_prntbits//F_bit)], m[0:int(F_sampling*N_prntbits//F_bit)])
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Original VCO output versus time')
    pl.grid(True)
    
    pl.subplot(3,1,2)
    pl.plot(t[0:int(F_sampling*N_prntbits//F_bit)], y[0:int(F_sampling*N_prntbits//F_bit)])
    pl.xlabel('Time (s)')
    pl.ylabel('Amplitude (V)')
    pl.title('Amplitude of carrier versus time')
    pl.grid(True)
    
    pl.subplot(3,1,3)
    pl.plot(f[0:int((F_carrier+F_dev*2)*N_FFT//F_sampling)], y_f[0:int((F_carrier+F_dev*2)*N_FFT//F_sampling)])
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Amplitude (dB)')
    pl.title('Spectrum')
    pl.grid(True)
    
    pl.tight_layout()
    pl.show()

"""
Data in
"""
# generate some random data for testing
data_in = np.random.randint(0, 2, N)

"""
VCO
"""
t = np.arange(0, float(N)/float(F_bit), 1/float(F_sampling), dtype=np.float64)
# extend the data_in to account for the bitrate and convert 0/1 to frequency
m = np.zeros(0).astype(float)
for bit in data_in:
    if bit == 0:
        m = np.hstack((m, np.multiply(np.ones(F_sampling//F_bit), F_carrier+F_dev)))
    else:
        m = np.hstack((m, np.multiply(np.ones(F_sampling//F_bit), F_carrier-F_dev)))

# calculate the output of the VCO
y = np.zeros(0)
y = A * np.cos(2*np.pi*np.multiply(m, t))
plot_data(y)

"""
Noisy Channel
"""
# create some noise
noise = (np.random.randn(len(y)) + 1) * A_n
snr = 10 * np.log10(np.mean(np.square(y)) / np.mean(np.square(noise)))
print(f"SNR = {snr:.2f} dB")

y = np.add(y, noise)
# view the data after adding noise
plot_data(y)

"""
Differentiator
"""
y_diff = np.diff(y, 1)

"""
Envelope detector + low-pass filter
"""
# create an envelope detector and then low-pass filter
y_env = np.abs(signal.hilbert(y_diff))
h = signal.firwin(numtaps=100, cutoff=F_bit*2, nyq=F_sampling/2)
y_filtered = signal.lfilter(h, 1.0, y_env)

# view the data after filtering
N_FFT = float(len(y_filtered))
f = np.arange(0, F_sampling/2, F_sampling/N_FFT)
w = np.hanning(len(y_filtered))
y_f = np.fft.fft(np.multiply(y_filtered, w))
y_f = 10 * np.log10(np.abs(y_f[0:int(N_FFT//2)]/N_FFT))

pl.subplot(3,1,1)
pl.plot(t[0:int(F_sampling*N_prntbits//F_bit)], m[0:int(F_sampling*N_prntbits//F_bit)])
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Original VCO output vs. time')
pl.grid(True)

pl.subplot(3,1,2)
pl.plot(t[0:int(F_sampling*N_prntbits//F_bit)], np.abs(y[0:int(F_sampling*N_prntbits//F_bit)]), 'b')
pl.plot(t[0:int(F_sampling*N_prntbits//F_bit)], y_filtered[0:int(F_sampling*N_prntbits//F_bit)], 'g', linewidth=3.0)
pl.xlabel('Time (s)')
pl.ylabel('Amplitude (V)')
pl.title('Filtered signal and unfiltered signal vs. time')
pl.grid(True)

pl.subplot(3,1,3)
pl.plot(f[0:int((F_carrier+F_dev*2)*N_FFT//F_sampling)], y_f[0:int((F_carrier+F_dev*2)*N_FFT//F_sampling)])
pl.xlabel('Frequency (Hz)')
pl.ylabel('Amplitude (dB)')
pl.title('Spectrum')
pl.grid(True)
pl.tight_layout()
pl.show()

"""
slicer
"""
# calculate the mean of the signal
mean = np.mean(y_filtered)
# if the mean of the bit period is higher than the mean, the data is a 0
rx_data = []
sampled_signal = y_filtered[F_sampling//F_bit//2:len(y_filtered):F_sampling//F_bit]

for bit in sampled_signal:
    if bit > mean:
        rx_data.append(0)
    else:
        rx_data.append(1)

bit_error = 0
for i in range(0, len(data_in)):
    if rx_data[i] != data_in[i]:
        bit_error += 1

print(f"bit errors = {bit_error}")
print(f"bit error percent = {float(bit_error)/float(N)*100:.2f}%")

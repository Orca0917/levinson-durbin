import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def levinson_durbin(p: int, y: np.ndarray):

    # pre-calculate the autocorrelation function R
    R = np.array([np.dot(y[i:], y[:len(y) - i]) for i in range(p + 1)])

    # initialize
    a = np.zeros(p)  # linear prediction coefficients
    k = np.zeros(p)  # reflection coefficients (PARCOR)
    E = R[0]         # initial prediction error

    # algorithm
    for i in range(p):

        # calculate reflection coeff (PARCOR)
        k[i] = (R[i + 1] - np.dot(a[:i], R[i:0:-1])) / E
        a[i] = k[i]

        # update coeffs
        if i > 0:
            a[:i] = a[:i] - k[i] * np.flip(a[:i])

        # update prediction error
        E = E * (1 - k[i] ** 2)
        
    return a, E


def plot_spectrum(a, E, sr=22050):
    a = np.concatenate(([1], -a))
    b = np.array([np.sqrt(E)])
    w, h = signal.freqz(b, a, worN=2048, fs=sr)

    plt.figure(figsize=(10, 6))
    plt.plot(w, 20 * np.log10(np.abs(h)), label='Magnitude Response', color='k')

    plt.xlim(0, sr/2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response of Linear Prediction Filter')
    plt.show()


def main():
    y, sr = librosa.load('LJ001-0001.wav', sr=22050)
    print(f"- sample rate: {sr} Hz")

    start_sample = 10000
    frame_size = 512

    y_frame = y[start_sample : start_sample + frame_size]
    y_frame = y_frame * np.hamming(frame_size)

    p = 12
    print(f"- prediction order (p): {p}, frame size: {len(y_frame)}")

    alpha, E = levinson_durbin(p, y_frame)

    print("\n## Levinson-Durbin Algorithm Results")
    print("---")
    print(f"- Linear prediction coefficients (alpha): {np.round(alpha, 2)}")
    print(f"- Prediction error (E): {np.round(E, 4)}")    

    plot_spectrum(alpha, E, sr)


if __name__ == '__main__':
    main()

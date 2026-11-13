"""
Signal Processing Using Python
A small practice project demonstrating signal generation, noise addition,
moving-average filtering, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(signal, window_size=15):
    """Smooth a signal using a simple moving-average filter."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")


def main():
    # Create a time axis and a clean example signal.
    sampling_rate = 500  # samples per second
    duration = 2          # seconds
    time = np.linspace(0, duration, sampling_rate * duration, endpoint=False)

    clean_signal = (
        np.sin(2 * np.pi * 5 * time)
        + 0.5 * np.sin(2 * np.pi * 12 * time)
    )

    # Add random noise to simulate a measured signal.
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0, scale=0.6, size=time.shape)
    noisy_signal = clean_signal + noise

    # Apply a moving-average filter to reduce high-frequency fluctuations.
    filtered_signal = moving_average(noisy_signal, window_size=15)

    # Plot the clean, noisy, and filtered signals for comparison.
    plt.figure(figsize=(10, 6))
    plt.plot(time, clean_signal, label="Original signal", linewidth=2)
    plt.plot(time, noisy_signal, label="Noisy signal", alpha=0.45)
    plt.plot(time, filtered_signal, label="Filtered signal", linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Signal Denoising with a Moving-Average Filter")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("signal_processing_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

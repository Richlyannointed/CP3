import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def main():
    histogram()

def histogram():
    data = np.loadtxt("Activity1Data.txt")[:, 1]
    binwidth = np.floor(np.std(data))/2
    bins = np.arange(np.floor(min(data)), np.floor(max(data)) + 1, binwidth)

    # Plot
    fig, ax = plt.subplots(1)
    ax.hist(data, bins, density=False, label="Data")
    ax.set_title("Histogram of 'Activity1Data.txt' Data")
    ax.set_xlabel("x")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.show()

    
if __name__ == "__main__":
    main()

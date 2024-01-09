import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def main():
    #make_do()
    histogram()

def histogram():
    random.seed(69420)
    data = np.loadtxt("Activity1Data.txt", skiprows=1)[:, 1]
    binwidth = np.std(data) / 2
    bins = np.arange(np.floor(min(data)), np.floor(max(data)) + 1, binwidth)

    mu = 40
    sigma = 2
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
    f = normal(x, mu=40, sigma=2) * binwidth * len(data)
    
    fig,ax = plt.subplots(1)
    ax.hist(data, bins, density=False, label="Data")
    ax.plot(x, f, "r-", label="Gaussian Fit")
    ax.set_title("Histogram of 'Activity1Data.txt' Data")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("x")
    ax.legend()
    fig.show()
 

def make_do():
    # No power, so I'll make my own 'Activity1Data.txt' file
    random.seed(69420)
    data = random.normal(40, 2, 60)
    with open('Activity1Data.txt', 'w') as outfile:
        outfile.write("File containing 60 random samples drawn from a normal distribution with a mean of 40.0 and std.dev.of 2.0")
        for i, x in enumerate(data):
            outfile.write(f"{i + 1}, {x:.1f}\n")

def normal(x, mu=0, sigma=1) -> float:
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
            
        
if __name__ == "__main__":
    main()

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
"""
The code, she estimate pi.
Also, make plot.

"""


def main():
    random.seed(4343)
    N = 50
    n = 100000
    pi, u_pi = measurement(N, n)
    print(f"From {N} independent Monte Carlo pi approximation experiments of {n} points each,\nwe estimate pi to be {pi:.5f} +/- {u_pi:.5f}\nAssuming a Type A uncertainty evaluation with a coverage probability of 68%")
    monte_pi_plot(31415)


# 3.3 Monte Carlo Determination of pi
def measurement(N: int, n: int) -> tuple:
    """
    Simulates N pi approximation experiments and reports a tuple of best estimate and standard uncertainty
    Assuming Type A uncertainty evaluation with a coverage probability of 68%
    N: number of experiments
    n: number of sampled points
    """
    experiments = [approximate_pi(n) for i in range(N)]
    pi = np.mean(experiments)
    u_pi = np.std(experiments) / np.sqrt(N)
    return (pi, u_pi)


def approximate_pi(n: int) -> float:
    """
    An attempt to vectorize the approximation of pi 
    using the Monte Carlo method given:
    n: number of points

    """
    x = random.uniform(-1, 1, n)
    y = random.uniform(-1, 1, n)
    def test(x, y):
        return x ** 2 + y **2 <= 1


    v=np.vectorize(test)
    circ = sum(v(x, y)) #using np.vectorize() Credit: https://stackoverflow.com/users/298607/dawg
    pi = 4 * circ / n
    return pi
    
    """
    # Slower Method
    in_circ = 0
    for i in range(n):
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        if x**2 + y**2 <= 1.0:
            in_circ += 1
        else:
            continue
    pi = 4 * in_circ / n
    return pi
    """

def monte_pi_plot(n: int) -> None:
    """
    For illustrative convenience, this only produces points in the first quadrant of the plane
    """
    x = random.uniform(-1, 0, n)
    y = random.uniform(-1, 0, n)
    def test(x, y):
        return x<=0 and y<=0 and x ** 2 + y **2 <= 1


    v=np.vectorize(test) #using np.vectorize() Credit: https://stackoverflow.com/users/298607/dawg
    circ = sum(v(x, y)) 
    pi = 4 * circ / n

    fig, ax= plt.subplots(1, figsize=(8, 8))
    
    # conditionally color the points
    col = np.where(x**2 + y**2 <=1, 'red', 'black' )
    
    ax.scatter(x, y, color=col, marker=".",s=4.5, label="Uniformly Sampled Points")
    ax.set_title(f"Single Monte Carlo Approximation Experiment:\n$\pi={pi:.3f}$, N={n} $(;$")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    fig.tight_layout()
    fig.savefig("monte_pi_plot.png")
    fig.show()


# 3.2 Monte Carlo Data Generation
def histogram():
    data = np.loadtxt("Activity1Data.txt")[:, 1]
    binwidth = np.std(data) / 2
    bins = np.arange(np.floor(min(data)), np.floor(max(data)) + 1, binwidth)

    mu = np.mean(data)
    sigma = np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    f = normal(x, mu=40, sigma=2) * binwidth * len(data)
    
    fig,ax = plt.subplots(1)
    ax.hist(data, bins, density=False, label="Data")
    ax.plot(x, f, "r-", label=f"Gaussian Fit\n$\\mu={mu:.1f}$\n$\\sigma={sigma:.1f}$")
    ax.set_title("Histogram of 'Activity1Data.txt' Data With $3\\sigma$ Gaussian")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("x")
    ax.legend()
    fig.tight_layout()
    #fig.savefig("hist_wgauss.png")
    fig.show()


def my_histogram():
    make_do()
    data = np.loadtxt("mydata.txt", skiprows=1)[:, 1]
    binwidth = np.std(data) / 2
    bins = np.arange(np.floor(min(data)), np.floor(max(data)) + 1, binwidth)

    mu = np.mean(data)
    sigma = np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    f = normal(x, mu=40, sigma=2) * binwidth * len(data)
    
    fig,ax = plt.subplots(1)
    ax.hist(data, bins, density=False, label="Data")
    ax.plot(x, f, "r-", label=f"Gaussian Fit\n$\\mu={mu:.1f}$\n$\\sigma={sigma:.1f}$")
    ax.set_title("Histogram of 'mydata.txt' Data With $3\\sigma$ Gaussian")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("x")
    ax.legend()
    fig.tight_layout()
    fig.savefig("my_hist_wgauss.png")
    fig.show()

def make_do():
    # No power, so I'll make my own 'Activity1Data.txt' file
    data = random.normal(40, 2, 60)
    with open('mydata.txt', 'w') as outfile:
        outfile.write("#File containing 60 random samples drawn from a normal distribution with a mean of 40.0 and std.dev.of 2.0")
        for i, x in enumerate(data):
            outfile.write(f"{i + 1} {x:.1f}\n")


def normal(x, mu=0, sigma=1) -> float:
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
            
        
if __name__ == "__main__":
    main()

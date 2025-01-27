import matplotlib.pyplot as plt
import numpy as np


file1 = "connected_disk_chain"
file2 = "non_overlapping_disks_N1000_L12"
file3 = "non_overlapping_disks_N2400_L12"
current = file3
dottxt = ".txt"
data = np.loadtxt(current + dottxt, delimiter='\t', dtype=float)
x, y = data[:, 0], data[:, 1]

r_disk = 0.1
L = 12  # System size (12x12 region)
N = len(x)


def compute_distances(x, y):
    distances = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            distances.append(dist)
    return np.array(distances)


# Compute all pairwise distances
distances = compute_distances(x, y)

# Define the radial bins for the pair-correlation function
r_max = L / 2
bins = np.linspace(r_disk, r_max, 100)

hist, edges = np.histogram(distances, bins=bins)

# Compute the radial distance of each bin's center
bin_centers = 0.5 * (edges[:-1] + edges[1:])

# Normalize the histogram to get g(r)
# Volume of each annular shell (2D area element): 2πr * dr
dr = np.diff(bins)
shell_areas = 2 * np.pi * bin_centers * dr  # area of each shell

density = N / (L**2)

g_r = hist / density

plt.plot(bin_centers, g_r, label="g(r)")
plt.xlabel('Distance r')
plt.ylabel('g(r)')
plt.title('Pair-correlation function')
plt.legend()
plt.savefig(current + ".png")
plt.show()

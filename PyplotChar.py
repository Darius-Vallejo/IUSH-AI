import matplotlib.pyplot as plt

"""
# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)

plt.show()

"""

class PLT:

    @classmethod
    def show(cls, any, title="Chart"):
        plt.figure()
        plt.plot(any)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.show()

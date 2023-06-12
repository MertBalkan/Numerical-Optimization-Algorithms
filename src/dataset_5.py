import numpy as np
import math
ti = np.arange(-4, 4, 0.4)
yi = [0.6294 * math.exp(0.8116 * t) + np.random.random() * 0.5 for t in ti] #np.random ile aldığımız gürültü 0.5 gürültü genliği gürültü 0.0 ise 0 hata ile sonuç bulur 0.8.. başına - koyarsak azalan olur
import matplotlib.pyplot as plt
plt.scatter(ti, yi, color="darkred")
plt.xlabel("ti")
plt.ylabel("yi")
plt.title("Data set 5")
plt.grid(color="green", linestyle="--", linewidth=0.1)
plt.show()
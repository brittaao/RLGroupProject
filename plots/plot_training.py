import matplotlib.pyplot as plt 
import pandas as pd

fig, ax = plt.subplots()

for i in range(1,8):
    data = pd.read_csv(f'train_results/model{i}.csv', header=None)

    data = data.transpose()
    ax.plot(range(1,1000,25), data)


plt.show()
#%%
import pandas as pd
data = pd.DataFrame(pd.read_csv('../metrics_analysis/losses_from_eval.txt', delimiter='|', header=None, engine='python'))
data.columns = ['Training Loss', 'Validation Loss', 'Dice', 'F3']
data.drop("Dice", axis=1, inplace=True)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
data.plot(xlabel = 'Epoch', ylabel ='Value', ax = ax)
plt.savefig('../metrics_analysis/loss_plot.pdf')
plt.show()
plt.close()

# %%

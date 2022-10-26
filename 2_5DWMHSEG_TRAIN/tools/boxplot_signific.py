import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

df = sns.load_dataset("tips")
x = "day"
y = "total_bill"
order = ['Sun', 'Thur', 'Fri', 'Sat']

# large figure size to show the legend
ax = sns.boxplot(data=df, x=x, y=y, order=order, width=0.4)

pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")]

annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
annotator.apply_and_annotate()
plt.tight_layout()
ax.figure.savefig("boxplot_signific.png")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('netflix_titles.csv')
data.head()

print('size of data: ',len(data))

type_cnt = data.type.value_counts()
type_cnt.values

# basic visualization
fig,ax = plt.subplots()
ax.bar(type_cnt.index, type_cnt.values)
ax.set_title('Number of movies and tv shows')
ax.set_ylabel('Count')
ax.set_xticks(type_cnt.index)
# ax.set_yticks(np.arange(0, 1000, 100))
plt.show()


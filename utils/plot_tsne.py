from sklearn import datasets
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:500]
y = digits.target[:500]

print (X.shape, y.shape)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

'''
0 -> formula_1
1 -> grass_cutting
2 -> water
3 -> helicopter
4 -> auto
5 -> cricket
6 -> guitar
7 -> sewing machine
8 -> stapler
9 -> traffic
'''

class_names = ['formula_1', 'grass_cutting', 'tap_water', 'helicopter', 'rikshaw', 'cricket', 'guitar', 'sewing', 'stapler', 'traffic']

target_ids = range(len(class_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, class_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()


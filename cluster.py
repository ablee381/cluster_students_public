from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from assessmentClass import *

resultFile = 'demo_test_results.xlsx'
results_obj = Results(resultFile)

if results_obj.type == 'Potts':
    results_obj.read_potts()
elif results_obj.type == 'Progress Learning':
    results_obj.read_norm_progress()
elif results_obj.type == 'Illuminate':
    results_obj.read_illuminate()
    results_obj.norm_illuminate()
else:
    assert False

questions = results_obj.data.columns.to_list()[1:]
standards = list(map(str, questions))
data = results_obj.data.loc[:, questions]

n = 4
kmeans = KMeans(n_clusters=n, init='random', random_state=1, n_init=10)
kmeans.fit(data)

pca = PCA(n_components=2)
pca.fit(data)
x = pca.transform(data)
components = np.abs(pca.components_)

clusters = np.array(kmeans.labels_)

results_obj.data.insert(loc=0, column='clusters', value=clusters)

# discrepancy between Progress Learning and everything gives key error
organized_data = results_obj.data.sort_values(by=['clusters', 'name'])
organized_data.to_csv(resultFile.split('.')[0] + '_CLUSTERED.csv')

cluster_means = results_obj.data.groupby(['clusters']).mean(numeric_only=True)
cluster_means.to_csv(resultFile.split('.')[0] + '_average_student.csv')

colors = ['b', 'r', 'k', 'orange']
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.set_xticks([])
ax.set_yticks([])
# Create labels based on the pca components
xy_labels = []
for i in range(2):
    labelParts = []
    important = sorted(components[i], reverse=True)[:3]
    for j in range(len(important)):
        tmpi = np.where(components[i] == important[j])[0][0]
        labelParts.append(standards[tmpi] + \
                          ' ' + \
                          str(np.round(pca.components_[i][tmpi], decimals=2)))
    label = '\n'.join(labelParts)
    xy_labels.append(label)

ax.set_xlabel(xy_labels[0], fontsize=15)
ax.set_ylabel(xy_labels[1], fontsize=15)

for i in range(n):
    tmpi = np.where(clusters == i)
    plt.scatter(x[tmpi][:, 0], x[tmpi][:, 1], c=colors[i])
plt.show()

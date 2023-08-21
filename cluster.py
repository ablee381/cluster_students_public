from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from assessmentClass import *
import dill


def cluster_students(resultFile, num_clusters=3):
    results_obj = Results(resultFile, guessRate=0.2)

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

    kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=1, n_init=10)
    kmeans.fit(data)
    clusters = np.array(kmeans.labels_)
    results_obj.data.insert(loc=0, column='clusters', value=clusters)

    # discrepancy between Progress Learning and everything gives key error
    organized_data = results_obj.data.sort_values(by=['clusters', 'name'])
    organized_data.to_csv(resultFile.split('.')[0] + '_CLUSTERED.csv')

    cluster_means = results_obj.data.groupby(['clusters']).mean(numeric_only=True)
    cluster_means.to_csv(resultFile.split('.')[0] + '_average_student.csv')
    return results_obj, data, standards, kmeans


def prescribe_topic(results_obj, standards, question_series):
    with open('topic_classifier.pkd', 'rb') as f:
        model = dill.load(f)
    topics = model.predict(question_series)
    if len(topics) != len(standards):
        topics = np.array(standards)

    unique_topics = np.unique(topics)
    class_avg = np.zeros(np.shape(unique_topics))
    class_stdev = np.zeros(np.shape(unique_topics))
    clusters = list(range(len(np.unique(results_obj.data['clusters']))))
    zscore = np.zeros((len(clusters),len(unique_topics)))
    # pull out the data by topic
    for i in range(len(unique_topics)):
        topic = unique_topics[i]
        topic_columns = [standards[j] for j in range(len(topics)) if topic == topics[j]]
        topic_columns.append('clusters')

        topic_df = results_obj.data[:, []].copy()
        class_avg[i] = topic_df[topic_columns[:-1]].mean(axis=None)
        class_stdev[i] = topic_df[topic_columns[:-1]].std(axis=None)
        for j in clusters:
            raw_mean = topic_df[topic_df['clusters'] == j, topic_columns[:-1]].mean(axis=None)
            zscore[j, i] = (raw_mean-class_avg)/class_stdev[i]
    topic_recommendation = []
    for c in clusters:
        tmpi = np.argmin(zscore[c, :])
        topic_recommendation.append(unique_topics[tmpi])
    to_df = {}
    to_df['group']=clusters
    to_df['topic']=topic_recommendation
    pd.DataFrame(to_df).to_csv(results_obj.resultSheet.split('.')[0]+ '_recommendations.csv')
    return topic_recommendation






def visualize_space(data, standards, kmeans, printPCA=False):
    pca = TruncatedSVD(n_components=2)
    pca.fit(data)
    x = pca.transform(data)
    components = np.abs(pca.components_)

    clusters = np.array(kmeans.labels_)

    colors = ['b', 'r', 'k', 'orange']
    fig = plt.figure(figsize=(7, 7))
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
            labelParts.append(standards[tmpi] +
                              ' ' +
                              str(np.round(pca.components_[i][tmpi], decimals=2)))
        label = '\n'.join(labelParts)
        xy_labels.append(label)

    ax.set_xlabel(xy_labels[0], fontsize=15)
    ax.set_ylabel(xy_labels[1], fontsize=15)

    for i in range(len(np.unique(clusters))):
        tmpi = np.where(clusters == i)
        plt.scatter(x[tmpi][:, 0], x[tmpi][:, 1], c=colors[i])

    if printPCA:
        fig.savefig(resultFile.split('.')[0] + '_PCA.png')
    else:
        plt.show()


if __name__ == '__main__':
    resultFile = 'past_use/Macecevic_3.xlsx'
    results_obj, data, standards, kmeans = cluster_students(resultFile)
    visualize_space(data, standards, kmeans)

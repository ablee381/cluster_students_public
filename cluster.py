from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from assessmentClass import *
import dill
from read_questions import *
from lang_process import *
import numpy as np


def cluster_students(resultFile, num_clusters=3, outDir='.'):
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
    organized_data.to_csv(outDir + '/' + resultFile.split('.')[0] + '_CLUSTERED.csv')

    cluster_means = results_obj.data.groupby(['clusters']).mean(numeric_only=True)
    cluster_means.to_csv(outDir + '/' + resultFile.split('.')[0] + '_average_student.csv')
    return results_obj, data, standards, kmeans


def prescribe_topic(results_obj, standards, question_series, outDir='.'):
    if question_series is None:
        topics = np.array(standards)
    else:
        with open('topic_classifier.pkd', 'rb') as f:
            model = dill.load(f)
        topics = model.predict(question_series)
        if len(topics) != len(standards):
            topics = np.array(standards)

    # make sure the data types of the column headers match with the standards
    results_obj.data.columns = results_obj.data.columns.astype(str)

    unique_topics = np.unique(topics)
    class_avg = np.zeros(np.shape(unique_topics))
    class_stdev = np.zeros(np.shape(unique_topics))
    clusters = list(range(len(np.unique(results_obj.data['clusters']))))
    zscore = np.zeros((len(clusters), len(unique_topics)))

    # pull out the data by topic
    for i in range(len(unique_topics)):
        topic = unique_topics[i]
        topic_columns = [standards[j] for j in range(len(topics)) if topic == topics[j]]
        topic_columns.append('clusters')

        topic_df = results_obj.data[topic_columns].copy()
        class_avg[i] = np.mean(topic_df[topic_columns[:-1]].stack())
        class_stdev[i] = np.std(topic_df[topic_columns[:-1]].stack())
        for j in clusters:
            raw_mean = np.mean(topic_df.loc[topic_df['clusters'] == j].stack())
            zscore[j, i] = (raw_mean-class_avg[i])/class_stdev[i]
    topic_recommendation = []
    for c in clusters:
        tmpi = np.argmin(zscore[c, :])
        topic_recommendation.append(unique_topics[tmpi])
    to_df = {'group': clusters, 'topic': topic_recommendation}
    pd.DataFrame(to_df).to_csv(outDir + '/' + results_obj.resultSheet.split('.')[0] + '_recommendations.csv')


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
    resultFile = 'demo/demo_test_results.xls'
    results_obj, data, standards, kmeans = cluster_students(resultFile)
    #visualize_space(data, standards, kmeans)
    file_str = pull_tex_from_zip('demo/unit_test.tex.zip')
    question_series = pd.Series(pull_test_w_questions(file_str))
    prescribe_topic(results_obj, standards, question_series)

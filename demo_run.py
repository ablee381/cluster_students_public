import pandas as pd
from cluster import *
from read_questions import pull_tex_from_zip, pull_test_w_questions
from lang_process import *



def main():
    results_obj, data, questions, kmeans = cluster_students('demo/demo_test_results.xls')
    file_str = pull_tex_from_zip('demo/unit_test.tex.zip')
    question_series = pd.Series(pull_test_w_questions(file_str))
    prescribe_topic(results_obj, questions, question_series)
    visualize_space(data, questions, kmeans)

if __name__ == '__main__':
    main()
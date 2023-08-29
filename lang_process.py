import dill
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import re

with open('standards_hierarchy.pkd', 'rb') as f:
    standards_df = dill.load(f)
# get probabilities of class from rf


def strip_white(s):
    return re.sub(r'\s+', ' ', s).strip()


def raw_data_and_labels(data_dict):
    """
    :param data_dict: two keys, the list of question text and the labels
    :return: list of question text and the labels
    """
    return pd.Series(data_dict['question_text']).apply(strip_white), \
        pd.Series(data_dict['standard'])


class SepTextEquations(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def find_equations(question):
        patterns = ['\[.+?\]', '\$.+?\$']
        eqn_list = []
        for pattern in patterns:
            eqn_list += re.findall(pattern, question)
        # for some reason re.findall is returning a list of tuples
        return eqn_list

    def pull_text(self, question):
        eqn_list = self.find_equations(question)
        # Delete all the equations from the question in reverse
        for eqn in eqn_list:
            start_ind = question.find(eqn)
            end_ind = start_ind + len(eqn)
            question = question[:start_ind] + question[end_ind:]
        return strip_white(question)

    def pull_equation(self, question):
        eqn_list = self.find_equations(question)
        return strip_white(' '.join(eqn_list))

    def transform(self, X):
        """
        :param X: pd.Series of text
        :return: pd.DataFrame w/columns: 'question_text', 'equation_text'
        """
        return pd.DataFrame({'question_text': X.apply(self.pull_text),
                             'equation_text': X.apply(self.pull_equation)})


class EquationTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def extract_symbol(eqn, symbol):
        """
        :param eqn: latex-style equation string
        :param symbol: single math symbol to be removed
        :return: list of the symbol repeated as many times as the symbol appeared, new string with the symbol removed.
        """
        eqn_parts = eqn.split(symbol)
        symbol_list = [symbol] * (len(eqn_parts) - 1)
        new_eqn = ' '.join(eqn_parts)
        return symbol_list, strip_white(new_eqn)

    @staticmethod
    def extract_pattern(eqn, pattern):
        """
        :param eqn:
        :param pattern:
        :return:
        """
        pattern_list = re.findall(pattern, eqn)
        new_eqn: str = strip_white(re.sub(pattern, ' ', eqn))
        return pattern_list, new_eqn

    @staticmethod
    def extract_graphic(eqn):
        pattern = r'\\includegraphics\[.*?\]\{.*?\}'
        graphic_list = re.findall(pattern, eqn)
        new_eqn = strip_white(re.sub(pattern, ' ', eqn))
        return len(graphic_list) * [r'\includegraphics'], new_eqn

    def eqn_tokenize(self, eqn):
        """
        :param eqn: latex-style equation string
        :return: list of tokens
        """
        # repeatedly make a call to a helper which removes a math symbol like + or - and puts the symbol in a list
        symbol_list = ['+', '-', '(', ')', '[', ']', '$', '=', ',']
        pattern_list = [r'\\[A-Z]?[a-z]+', r'\w\^{.+?}', r'\^{?\w+}?', r'\w_{.+?}', r'\d+\.\d+']
        # pull out symbols that would have messed with regex patterns
        second_symbol_list = ['.', '{', '}', '\\']
        token_list = []
        # extract the graphics because they can mess with some of the regex later
        tmp_list, eqn = self.extract_graphic(eqn)
        token_list.extend(tmp_list)
        for symbol in symbol_list:
            tmp_list, eqn = self.extract_symbol(eqn, symbol)
            token_list.extend(tmp_list)
        for pattern in pattern_list:
            tmp_list, eqn = self.extract_pattern(eqn, pattern)
            token_list.extend(tmp_list)
        for symbol in second_symbol_list:
            tmp_list, eqn = self.extract_symbol(eqn, symbol)
            token_list.extend(tmp_list)
        token_list.extend(eqn.split(' '))
        return ' '.join(token_list)

    def transform(self, X):
        """
        :param X: Series of math strings
        :return: Series of List of string tokens
        """
        return X.apply(lambda x: self.eqn_tokenize(x))


class TokenColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_tokenizer, eqn_tokenizer, text_ind=0, eqn_ind=1):
        self.text_tokenizer = text_tokenizer
        self.eqn_tokenizer = eqn_tokenizer
        self.text_ind = text_ind
        self.eqn_ind = eqn_ind

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: pd.DataFrame where column 0 is the text series and column 1 is the eqn series
        :return: Single series of spacy tokens
        """
        text_series = X[X.columns[self.text_ind]]
        eqn_series = X[X.columns[self.eqn_ind]]
        eqn_token_str_series = self.eqn_tokenizer.transform(eqn_series)
        tmp_series = text_series + ' ' + eqn_token_str_series
        return tmp_series


class MathClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_features, n_estimators, max_depth):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.pipe = Pipeline([
            ('tokenize', EquationTokenizer()),
            ('tf-idf', TfidfVectorizer(max_features=self.max_features,
                                       tokenizer=lambda x: x.split(' '),
                                       min_df=2)),
            ('classify', RandomForestClassifier(n_estimators=self.n_estimators,
                                                max_depth=self.max_depth))
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)


def automate_linear_gs(mdl, gs_dict):
    for key in gs_dict:
        distributions = {key: range(gs_dict[key]['min'],
                                    gs_dict[key]['max'],
                                    gs_dict[key]['step'])}

        gs = GridSearchCV(
            mdl,
            distributions,
            cv=5,
            verbose=2
        )
        gs.fit(X_rand, y_rand)
        f = open('math_classifier_hyperparameters.txt', 'a')
        f.write('Tune parameters: ' + str(distributions) + '\n')
        f.write(key + ': ' + str(gs.best_params_[key]) + '\n')
        f.close()


def automate_true_gs(mdl, gs_dict):
    distributions = {}
    for key in gs_dict:
        distributions[key] = range(gs_dict[key]['min'],
                                   gs_dict[key]['max'],
                                   gs_dict[key]['step'])

    gs = GridSearchCV(
        mdl,
        distributions,
        cv=5,
        verbose=2
    )
    gs.fit(X_rand, y_rand)
    f = open('math_classifier_hyperparameters.txt', 'a')
    f.write('Tune parameters: ' + str(distributions) + '\n')
    for key in distributions:
        f.write(key + ': ' + str(gs.best_params_[key]) + '\n')
    f.close()


def analyze_cross_val(cv_results, cv, X, y, standards_df):
    """
    :param cv_results: Assumes the following are true:
        return_estimators = True
        cv is set to a specified split which is passed in as cv
    :param cv: the split object that determines the train-test-split used in cross validation
    :return: array of dfs of the real label and the missed classification label, one for each fold
    """
    test_ind_list = [split[1] for split in cv.split(X, y)]
    out = []
    for i in range(len(test_ind_list)):
        mdl = cv_results['estimator'][i]
        test_ind = test_ind_list[i]
        y_test = y[test_ind]
        y_pred = mdl.predict(X[test_ind])
        tmpi = y_test != y_pred
        y_truth = y_test[tmpi]
        y_miss = y_pred[tmpi]
        topic_truth = []
        topic_miss = []
        unit_truth = []
        unit_miss = []
        for j in range(len(y_miss)):
            standard_truth = y_truth[y_truth.index[j]].split('_')[0]
            standard_miss = y_miss[j].split('_')[0]
            topic_truth.append(
                max(standards_df.loc[standards_df['Standard'] == standard_truth]['Topic']))
            topic_miss.append(
                max(standards_df.loc[standards_df['Standard'] == standard_miss]['Topic']))
            unit_truth.append(
                list(standards_df.loc[standards_df['Standard'] == standard_truth]['Unit']))
            unit_miss.append(
                list(standards_df.loc[standards_df['Standard'] == standard_miss]['Unit']))
        df = pd.DataFrame({'truth': y_truth, 'mislabel': y_miss,
                           'topic_truth': topic_truth, 'topic_miss': topic_miss,
                           'unit_truth': unit_truth, 'unit_miss': unit_miss})
        out.append(df)
    return out


def convert_label(standard, standards_df):
    """
    :param standard: string, original label for the question
    :param standards_df: contains information on the standard's topic and units where it appears
    :return: the topic covered by the standard
    """
    y = standard.split('_')[0]
    topic = max(standards_df.loc[standards_df['Standard'] == y]['Topic'])
    return topic


if __name__ == '__main__':
    with open('training_data.pkd', 'rb') as f:
        data_dict = dill.load(f)
    raw_data, standards = raw_data_and_labels(data_dict)
    topics = [convert_label(standard, standards_df) for standard in standards]
    print(len(set(topics)))
    X_rand, y_rand, standard_rand = shuffle(raw_data, topics, standards, random_state=17)
    math_classifier = MathClassifier(max_features=340,
                                     n_estimators=400,
                                     max_depth=12)
    math_classifier.fit(X_rand, y_rand)
    with open('topic_classifier.pkd', 'wb') as f:
        dill.dump(math_classifier, f)
    # cv = StratifiedKFold(n_splits=5)
    # cv_results = cross_validate(math_classifier,
    #                            X_rand, y_rand,
    #                            cv=cv,
    #                            return_train_score=True,
    #                            return_estimator=True)

    # miss_list = analyze_cross_val(cv_results, cv, X_rand, standard_rand, standards_df)

    # for d in miss_list:
    #    print(d)

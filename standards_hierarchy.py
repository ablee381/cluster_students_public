# Creates a hierarchical structure for the 56 standards. This is all hardcoded by hand.

import pandas as pd
import dill

standards_df = pd.read_csv('standards_topics.csv')
standards_df['Standard'] = standards_df.loc[:, 'Standard'].apply(lambda x: x.strip())
standards_df['Topic'] = standards_df.loc[:, 'Topic'].apply(lambda x: x.strip())


if __name__ == '__main__':
    # The following code tests that standards and topics are 1:1.
    # Note that standards and units are not 1:1, there are a few standards that show up in
    # multiple units

    with open('training_data.pkd', 'rb') as f:
        data_dict = dill.load(f)
    labels = data_dict['standard']
    missing_labels = set([])
    for label in labels:
        y = label.split('_')[0]
        if len(standards_df.loc[standards_df['Standard']==y])==0:
            missing_labels.add(label)
    print(len(missing_labels))
    print(missing_labels)
    print(0)

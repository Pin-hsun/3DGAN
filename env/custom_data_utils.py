import pandas as pd

def customize_data_split(dataset, split=None):
    if split is not None:
        folder = '/full/'
        if dataset == 'womac3':
            if split == 'moaks':
                df = pd.read_csv('env/subjects_unipain_womac3.csv')
                train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
                test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
            elif split == 'a':
                train_index = range(213, 710)
                test_index = range(0, 213)
            elif split == 'b':
                train_index = range(0, 497)
                test_index = range(497, 710)
    else:
        folder = '/train/'
        train_index = None
        test_index = None
    return folder, train_index, test_index
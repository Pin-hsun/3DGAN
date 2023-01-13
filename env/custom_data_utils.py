import pandas as pd


def customize_data_split(args):
    dataset = args.dataset
    split = args.split
    if split is not None:
        folder = '/full/'
        if dataset == 'womac3':
            if split == 'moaks':
                df = pd.read_csv('env/subjects_womac3.csv')
                train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
                test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
            elif split == 'a':
                train_index = range(213, 710)
                test_index = range(0, 213)
            elif split == 'b':
                train_index = range(0, 497)
                test_index = range(497, 710)
            elif split == 'small':
                train_index = range(690, 710)
                test_index = range(0, 20)

        if dataset == 'womac4':
            if split == 'b':
                train_index = range(666, 2225)
                test_index = range(0, 666)

        if dataset == 'oaiseg':
            if split == 'a':
                if args.load3d:
                    train_index = range(0, 70)
                    test_index = range(70, 88)
                else:
                    train_index = range(2155, 9924) # 7769
                    test_index = range(0, 2155)

        if dataset == 't2d':
            if split == 'a':
                train_index = range(0, 14000)
                test_index = range(14000, 16317)

        if dataset == '40xmun':
            if split == 'a':
                train_index = range(0, 8000)
                test_index = range(8000, 8160)

    else:
        folder = '/train/'
        train_index = None
        test_index = None
    return folder, train_index, test_index
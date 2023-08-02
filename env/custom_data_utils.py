import pandas as pd
import os, glob


def customize_data_split(args):
    dataset = args.dataset
    split = args.split
    if split is not None:
        folder = '/full/'
        first_folder = args.direction.split('%')[0].split('_')[0]
        length_data = len(sorted(glob.glob(os.path.join(os.environ.get('DATASET') + args.dataset + folder, first_folder, '*'))))
        #print(os.path.join(os.environ.get('DATASET') + args.dataset + folder, args.direction[0], '*'))

        if split in ['x', 'y', 'small', 'all']:
            if split == 'x':
                train_index = range(0, length_data // 10 * 7)
                test_index = range(length_data // 10 * 7, length_data)
            if split == 'y':
                train_index = range(length_data // 10 * 3, length_data)
                test_index = range(0, length_data // 10 * 3)
            if split == 'small':
                train_index = range(0, length_data // 10)
                test_index = range(length_data // 10, length_data)
            if split == 'all':
                train_index = range(0, length_data)
                test_index = range(0, 1)
        else:  # THIS IS WRONG FOR LOAD3D ???
            if dataset == 'womac3':
                if split == 'moaks':
                    df = pd.read_csv('env/csv/womac3.csv')
                    train_index = [x for x in range(df.shape[0]) if not df['has_moaks'][x]]
                    test_index = [x for x in range(df.shape[0]) if df['has_moaks'][x]]
                elif split == 'a':
                    train_index = range(213, 710)
                    test_index = range(0, 213)
                elif split == 'b':
                    train_index = range(0, 497)
                    test_index = range(497, 710)
            if dataset == 'womac4':
                if split == 'a':
                    train_index = range(667, 2225)
                    test_index = range(0, 667)
            if dataset == 'oaiseg':
                if split == 'a':
                    if args.load3d:
                        train_index = range(0, 70)
                        test_index = range(70, 88)
                    else:
                        train_index = range(2155, 9924) # 7769
                        test_index = range(0, 2155)
    else:
        folder = '/train/'
        train_index = None
        test_index = None
    return folder, train_index, test_index
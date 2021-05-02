import pandas as pd
import argparse
import os

def choose_labels(class_id_df, n_classes, random_state=420):
    """ Chooses n_classes from all classes in class_id_df.

        Returns:
            class_id_df - dataframe of class ids
            n_classes - number of classes to filter
    """
    class_id_df =  class_id_df.sample(n_classes, random_state=random_state)
    return class_id_df, class_id_df["ClassId"].to_list()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decrease number of classes to a chosen size.')
    parser.add_argument('--n_classes', 
                        type=int, 
                        help='number to classes to filter to', 
                        required=True)
    parser.add_argument('--dataset_dir', nargs='?', default='dataset', 
                        help='dataset directory')
    parser.add_argument('--output_dir', nargs='?', default='data', 
                        help='output directory for filtered csvs')
    args = parser.parse_args()

    # make new directory for outputs if it does not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # read csv with class ids
    class_df = pd.read_csv(f'{args.dataset_dir}/SignList_ClassId_TR_EN.csv')
    u_len_label = len(class_df['ClassId'].unique())
    print("total unique label:", u_len_label)

    # check if no. of classes to sample is valid
    if args.n_classes < 1 or args.n_classes > u_len_label:
        raise ValueError(f"please use an int which is between 1 and the number of all classes ({u_len_label})")
    
    # extract sampled classes and save
    class_df, sampled_class_ids = choose_labels(class_df, args.n_classes)
    class_df = class_df.reset_index(drop=True)
    class_df["oldClassId"] = class_df["ClassId"]
    class_df["ClassId"] = class_df.index
    class_df.to_csv(f"{args.output_dir}/filtered_ClassId.csv", index=False)

    # read csv from dataset
    df_store = {}
    df_store["train"] = pd.read_csv(f'{args.dataset_dir}/train_labels.csv', header=None)
    df_store["val"] = pd.read_csv(f'{args.dataset_dir}/val_ground_truth.csv', header=None)
    df_store["test"] = pd.read_csv(f'{args.dataset_dir}/test_ground_truth.csv', header=None)

    # get filtered sets and save them
    for dataset_type, df in df_store.items():
        df = df[df[1].isin(sampled_class_ids)]
        df[1] = df[1].map(dict(zip(class_df["oldClassId"], class_df["ClassId"])))
        print(f"filtered {dataset_type} set has {len(df)} rows")
        df.to_csv(f"{args.output_dir}/{dataset_type}.csv", header=False, index=False)


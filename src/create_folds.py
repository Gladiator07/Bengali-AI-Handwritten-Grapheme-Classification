import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CSV)
    print(df.head())

    df.loc[:, 'kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_id.values
    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(mskf.split(X, y)):
        print("TRAIN: ", trn_, "VAL: ", val_)
        df.loc[val_, "kfold"] = fold
    
    print(df.kfold.value_counts())

    df.to_csv(config.TRAIN_FOLDS_CSV)
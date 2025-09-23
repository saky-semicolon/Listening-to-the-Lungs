import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import json

class CFG:
    sample_rate = 16000
    n_mels = 128
    work_dir = "work_tf"
    classes = ("Bronchial", "asthma", "copd", "healthy", "pneumonia")
    test_size = 0.15
    val_size = 0.15
    seed = 42
    # training defaults
    epochs = 10
    early_stop_pat = 6

def prepare_splits(df_all):
    df_trainval, df_test = train_test_split(
        df_all, test_size=CFG.test_size,
        stratify=df_all["label_id"], random_state=CFG.seed
    )
    val_rel = CFG.val_size / (1.0 - CFG.test_size)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_rel,
        stratify=df_trainval["label_id"], random_state=CFG.seed
    )

    os.makedirs(CFG.work_dir, exist_ok=True)
    df_train.to_csv(f"{CFG.work_dir}/train.csv", index=False)
    df_val.to_csv(f"{CFG.work_dir}/val.csv", index=False)
    df_test.to_csv(f"{CFG.work_dir}/test.csv", index=False)

    label2id = {c: i for i, c in enumerate(CFG.classes)}
    with open(f"{CFG.work_dir}/labels.json", "w") as f:
        json.dump({"label2id": label2id}, f, indent=2)

    return df_train, df_val, df_test

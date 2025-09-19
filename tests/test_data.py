import pandas as pd
import os
from src.data import prepare_splits, CFG

def test_prepare_splits(tmp_path):
    # create tiny df_all
    classes = list(CFG.classes)
    rows = []
    for i in range(30):
        lbl = classes[i % len(classes)]
        rows.append({'path': f"file_{i}.wav", 'label': lbl, 'label_id': classes.index(lbl)})
    df = pd.DataFrame(rows)
    # set work_dir to tmp
    CFG.work_dir = str(tmp_path / "work_tf_test")
    df_train, df_val, df_test = prepare_splits(df)
    # files created?
    assert os.path.exists(os.path.join(CFG.work_dir, "train.csv"))
    assert os.path.exists(os.path.join(CFG.work_dir, "val.csv"))
    assert os.path.exists(os.path.join(CFG.work_dir, "test.csv"))

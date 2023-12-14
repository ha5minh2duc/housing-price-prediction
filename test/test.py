

import joblib
import numpy as np
import pandas as pd
from src.final import FINAL_FEATURES_NAMES, PATH_FINAL_MODEL
from time import time




if __name__ == "__main__":
    s1 = time()
    model = joblib.load(PATH_FINAL_MODEL)
    s2 = time()
    t = [91, 2, 'TRAN-PHU', 148873,
         149779, 195401, 'VAN-QUAN', 'MISSING'] # 29935546
    dt = pd.DataFrame(columns=FINAL_FEATURES_NAMES)
    dt.loc[0] = t
    print(dt)
    pred = model.predict(dt)
    s3 = time()
    print(f"predict={np.round(pred[0])}, s2-s1={s2-s1}, s3-s2={s3-s2}")

    s2 = time()
    t = [str(c) for c in t]
    print(t)

    dt = pd.DataFrame(columns=FINAL_FEATURES_NAMES)
    dt.loc[0] = t
    print(dt)
    pred = model.predict(dt)
    s3 = time()
    print(f"predict={np.round(pred[0])}, s2-s1={s2-s1}, s3-s2={s3-s2}")


    def func(x):
        try:
            return float(x)
        except:
            return x
    s2 = time()
    t = [func(c) for c in t]
    print(t)

    dt = pd.DataFrame(columns=FINAL_FEATURES_NAMES)
    dt.loc[0] = t
    print(dt)
    pred = model.predict(dt)
    s3 = time()
    print(f"predict={np.round(pred[0])}, s2-s1={s2 - s1}, s3-s2={s3 - s2}")

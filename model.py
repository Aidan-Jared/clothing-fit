import numpy as np
import h2o
h2o.init()
import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("data/modcloth_final_data.json", lines=True)
    fit = h2o.H2OFrame(df)
    print(fit)
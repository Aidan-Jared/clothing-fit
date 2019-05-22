import numpy as np
import h2o
h2o.init()
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
# from sklearn.preprocessing import Imputer

if __name__ == "__main__":
    df = pd.read_json("data/modcloth_final_data.json", lines=True)
    df.drop(["review_summary", "review_text", "user_name", "quality"], axis=1, inplace=True)
    fit = h2o.H2OFrame(df, na_strings=['NA', 'none', 'nan', "NaN"])
    factor_col = ["fit", "bust", "category", "height", "length", "shoe size", "shoe width", "waist"]
    for i in factor_col:
        fit[i] = fit[i].asfactor()
    y = "fit"
    x = fit.columns
    del x[4]
    del x[11]
    train, valid, text = fit.split_frame([.6,.2], seed=50)
    params = {
        'ntrees' : [10,20,30,40],
        'nbins_cats': [2,4,8,16,32],
        'learn_rate' : [1,.1,.01],
        'sample_rate' : [.9,.8],
        'col_sample_rate' : [.9,.8],
        'seed' : [42],
        'stopping_rounds' : [5],
        'stopping_tolerance' : [1e-5],
        'stopping_metric' : ['logloss'],
        'balance_classes': [True, False]}
    grid = H2OGridSearch(H2OGradientBoostingEstimator, params, grid_id='gbm_grid_fit1')
    grid.train(x=x, y=y, training_frame=train, validation_frame=valid)
    print(grid)
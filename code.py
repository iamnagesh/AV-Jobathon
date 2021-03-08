import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


import lightgbm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

########################### functions start ###########################

def xdf_unpack(x):
    xdf_respose = x.pop("Response")
    x.insert(len(x.columns), "Response", xdf_respose)
    tr = x.copy().loc[x['Response'] != -1]
    te = x.copy().loc[x['Response'] == -1]
    te.drop(["Response"], axis= 1, inplace = True)
    return tr, te

def xgb_model(train, test, fname, cv = 5, met = 'auc'):
    val = np.zeros(train.shape[0])
    pred = np.zeros(test.shape[0])
    x = train.drop(["ID","Response"],axis=1).values
    y = train["Response"].values
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1234)
    model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
          colsample_bynode=1, colsample_bytree=0.8, gamma=1,
          learning_rate=0.02, max_delta_step=0, max_depth=5,
          min_child_weight=5, missing=None, n_estimators=600, n_jobs=1,
          nthread=1, objective='binary:logistic', random_state=0, reg_alpha=0,
          reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
          subsample=0.8, verbosity=1)

    # training and pred in folds
    for fold_index, (train_index,val_index) in enumerate(folds.split(x,y)):
        print('Batch {} started...'.format(fold_index))
        gc.collect()
        bst = model_xgb.fit(x[train_index],y[train_index],
                  eval_set = [(x[val_index],y[val_index])],
                  early_stopping_rounds=200,
                  verbose= 200,
                  eval_metric ='auc'
                  )
        val[val_index] = model_xgb.predict_proba(x[val_index])[:,1]
        val_acc = metrics.roc_auc_score(y[val_index],val[val_index])
        print('auc of this val set is {}'.format(val_acc))
        pred += model_xgb.predict_proba(test.drop(["ID"],axis=1).values)[:,1]/folds.n_splits
    df_sub = pd.read_csv("./input/sample_submission_QrCyCoT.csv")
    df_sub.Response = pred
    dt = datetime.now().strftime("%d-%m_%H_%M")
    str2 = "xgb__%s__CV__%s__%s__%s__%s"%(cv, met,  round(val_acc,3), fname, dt)
    df_sub.to_csv(str2, index = False)
    return model_xgb, pred, str2

def lgbm_model(x, categorical_features, remove_features = None):
    import warnings
    warnings.filterwarnings("ignore")

    lgbm_y = x.Response.values
    if remove_features != None:
        lgbm_x = x.drop(['ID', 'Response']+ remove_features, axis=1)
    else:
        lgbm_x = x.drop(['ID', 'Response'], axis=1)

    lgb_x, lgb_x_val, lgb_y, lgb_y_val = train_test_split(lgbm_x, lgbm_y, test_size=0.2, random_state=42, stratify=lgbm_y)

    len(lgb_y)+len(lgb_y_val)
    # categorical_features = ["CC", "RC", "Acc_Type", "Reco_Ins_Type", "Is_Spouse",
    #                         "Hth_Ind", "Hold_Pol_Type", "Reco_Pol_Cat", "Gen_CAT"]
    if remove_features != None:
        categorical_features = set(categorical_features) -set(remove_features)
    lgbm_train = lightgbm.Dataset(lgb_x, label=lgb_y, categorical_feature=categorical_features)
    lgbm_val = lightgbm.Dataset(lgb_x_val, label=lgb_y_val)

    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'verbose': 0,
        'colsample_bytree': 0.95,
        'min_child_samples': 243,
        'min_child_weight': 1,
        'num_leaves': 25,
        'reg_alpha': 10,
        'reg_lambda': 0.1,
        'subsample': 0.46
    }
    lgb_cv = lightgbm.cv(parameters,
                           lgbm_train,
                           nfold = 5,
                           num_boost_round=5000,
                           stratified = True,
                           early_stopping_rounds=50)

    print("lightgbm cross validation accuracy for 5 folds:", max(lgb_cv['auc-mean']))

    lgbm_train = lightgbm.Dataset(lgb_x, label=lgb_y, categorical_feature=categorical_features)
    lgbm_val = lightgbm.Dataset(lgb_x_val, label=lgb_y_val)
    model = lightgbm.train(parameters,
                           lgbm_train,
                           valid_sets=lgbm_val,
                           num_boost_round=5000,
                           verbose_eval = 100,
                           early_stopping_rounds=50)
    return model

def xgb_feature_imp(xgb, feat_names, filename="xgb_default"):
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    sorted_idx = xgb.feature_importances_.argsort()
    plt.barh(feat_names[sorted_idx], xgb.feature_importances_[sorted_idx])
    plt.xlabel("Xgboost Feature Importance")
    plt.savefig("./models/"+filename+".png")

def xdf_pack(xtrain, xtest):
    xtest["Response"] = -1
    xdf = xtrain.append(xtest, ignore_index = True)
    print('xdf (rows,cols):',xdf.shape)
    return xdf
########################### functions End ###########################

print('Current Directory: ',os.getcwd())
path = "D:\\AV\\Jobathon"
os.chdir(path)
print('New Directory: ',path)
# from src.custom_functions import *
gc.enable()


TRAINING_input = "./input/train_Df64byy.csv"
TESTING_input = "./input/test_YCcRUnU.csv"
train = pd.read_csv(TRAINING_input)
test = pd.read_csv(TESTING_input)

new_column_names = ['ID', 'CC', 'RC', 'Acc_Type',
       'Reco_Ins_Type', 'UAge', 'LAge', 'Is_Spouse',
       'Hth_Ind', 'Hold_Pol_Dur', 'Hold_Pol_Type',
       'Reco_Pol_Cat', 'Reco_Pol_Pre']
train.columns = new_column_names + train.columns.tolist()[len(new_column_names):]
test.columns = new_column_names

print('train (rows,cols):',train.shape)
print('test (rows,cols):',test.shape)

xdf = xdf_pack(train, test)

########################### feature engineering 1 start ###########################

# Preserving potential ordinal features
xdf['Hold_Pol_Dur'] = xdf['Hold_Pol_Dur'].astype(str)
xdf['Hold_Pol_Dur'] = xdf['Hold_Pol_Dur'].replace({"14+": "15", "nan":"0"}).astype(float).astype(int)
xdf['Hold_Pol_Type'] = xdf['Hold_Pol_Type'].astype(str)
xdf['Hold_Pol_Type'] = xdf['Hold_Pol_Type'].replace({"nan":"0"}).astype(float).astype(int)
xdf['CC'] = xdf['CC'].str.replace("C", "")
xdf['CC'] = xdf['CC'].astype(int)
xdf['Hth_Ind'] = xdf['Hth_Ind'].str.replace("X", "")
xdf['Hth_Ind'] = xdf['Hth_Ind'].fillna(0).astype(int)

# vanilla LabelEncoding
xdf['Acc_Type'] = LabelEncoder().fit_transform(xdf.Acc_Type)
xdf['Is_Spouse'] = LabelEncoder().fit_transform(xdf.Is_Spouse)
xdf['Reco_Ins_Type'] = LabelEncoder().fit_transform(xdf.Reco_Ins_Type)

train, test = xdf_unpack(xdf)
########################### feature engineering 1 end ###########################

'''
lightgbm has performed better than xgboost probably due to a large number
of categorical variables in the data which can be handled by lightgbm

side note: xgboost model hyperparameters have been optimized on data after
feature engineering 1, where as no efforts have been put in selecting lightgbm's hyperparameters

The following are the two best submissions, removed the rest to make this short and uncluttered
'''

'''
lgbm 6:
val auc: 0.706
public LB: 0.707
private LB: 0.695
'''
cat_feat = ["CC", "RC", "Acc_Type", "Reco_Ins_Type", "Is_Spouse",
                        "Hth_Ind", "Hold_Pol_Type", "Reco_Pol_Cat"]
lgbm_6 = lgbm_model(train, categorical_features= cat_feat)
lightgbm.plot_importance(lgbm_6, figsize=(16, 16), dpi=80)
lgbm_6_pred = lgbm_6.predict(test.drop('ID', axis=1))
# df_sub.Response = lgbm_6_pred
# df_sub.to_csv("./output/lgbm_6_5_CV_0.706_base_model_labelencoding_28-02_14_18.csv", index=False)

########################### feature engineering 2 start ###########################

xdf = xdf_pack(train, test)
xdf['premium_gen'] = round((xdf['UAge'] - xdf['LAge'])/18,2)

# Generation categorical
xdf['Gen_CAT'] = 0
xdf.loc[(xdf['premium_gen'] >0) & (xdf['premium_gen'] <=0.5), 'Gen_CAT'] = 1
xdf.loc[(xdf['premium_gen'] >0.5) & (xdf['premium_gen'] <=1), 'Gen_CAT'] = 2
xdf.loc[(xdf['premium_gen'] >1) & (xdf['premium_gen'] <=2), 'Gen_CAT'] = 3
xdf.loc[(xdf['premium_gen'] >2), 'Gen_CAT'] = 4

# premium per generation
xdf['premium_gen'] = xdf["premium_gen"] * xdf["Reco_Pol_Pre"]

freq_enc_vars = ['CC', 'RC', 'Hth_Ind', 'Hold_Pol_Type', 'Reco_Pol_Cat', 'Hold_Pol_Dur']
for i in freq_enc_vars:
    f = xdf.groupby(i).size()
    fe = f/len(xdf)*100
    xdf.loc[:,i+"_fe_perc"] = xdf[i].map(fe)
    xdf.loc[:,i+"_fe"] = xdf[i].map(f)

train, test = xdf_unpack(xdf)

########################### feature engineering 2 end ###########################


'''
lgbm 21:
val auc: 0.728
public LB: 0.726
private LB: 0.711
'''
cat_feat = ["CC", "RC", "Acc_Type", "Reco_Ins_Type", "Is_Spouse",
                        "Hth_Ind", "Hold_Pol_Type", "Reco_Pol_Cat", "Gen_CAT"]
remove_features= ["LAge", "premium_gen", "Hth_Ind", "Reco_Pol_Pre"]
lgbm_21 = lgbm_model(train, categorical_features = cat_feat, remove_features= remove_features)
lightgbm.plot_importance(lgbm_21)
lgbm_21_pred = lgbm_21.predict(test.drop(['ID']+remove_features, axis=1))


'''
Final submission:
average of lgbm_21_pred and lgbm_6_pred
public LB: 0.720
private LB: 0.706
'''

final_sub = (lgbm_21_pred + lgbm_6_pred)/2

'''
Lessons learnt:
1. Catboost classifier would work really well out of the box without any major tweaks
2. Should try all the possible models on the baseline data so that I can get a brief idea of each models capabilities on the Dataset

Below, catboost has been tried and it gives a eval auc of 0.81, when finetuned/stacked or ensembled could result in better auc


from catboost import CatBoostClassifier, Pool

def catboost_model(data, remove_features=[], cat_feat = []):
    _y = data.Response.values
    if remove_features != None:
        _x = data.drop(['ID', 'Response']+ remove_features, axis=1)
    else:
        _x = data.drop(['ID', 'Response'], axis=1)
    x, x_val, y, y_val = train_test_split(_x, _y, test_size=0.2, random_state=42, stratify= _y)
    params = {
        'cat_features': cat_feat,
        'eval_metric': 'AUC',
        'random_seed': 1234,
        'n_estimators': 1000,
    }
    cb = CatBoostClassifier(**params, early_stopping_rounds=200, cat_features=cat_feat)
    cb_model = cb.fit(x, y, eval_set=(x_val,y_val), plot=True, verbose=True)
    return cb_model

cat_feat = ["CC", "RC", "Acc_Type", "Reco_Ins_Type", "Is_Spouse",
                        "Hth_Ind", "Hold_Pol_Type", "Reco_Pol_Cat", "Gen_CAT"]
cb_1 = catboost_model(train, cat_feat)
cb_1
'''

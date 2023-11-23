import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sMAPE(actual, predicted):
    return np.mean(200* (np.abs(actual-predicted) / (np.abs(actual)+np.abs(predicted))))
    
def run_model(train_df, valid_df, si_scores_df, scores_df, f, future=False, **kwargs): 
    res_train = train_df.copy()
    res_valid = valid_df.copy()
    
    name = kwargs['name']
    res_train[name] = pd.Series()
    res_valid[name] = pd.Series()
    
    for i in range(50):
        for s in range(10):            
            # f function trains and fits model on valid data to produce predictions for valid set
            train = train_df.loc[(train_df.store == s+1) & (train_df.item == i+1)]
            valid = valid_df.loc[(valid_df.store == s+1) & (valid_df.item == i+1)]
            preds_train, preds_valid = f(train, valid, **kwargs['f_kwargs'])

            # store in output df
            res_train.loc[(res_train.store == s+1) & (res_train.item == i+1), name] = preds_train
            if not future:
                res_valid.loc[(res_valid.store == s+1) & (res_valid.item == i+1), name] = preds_valid
            else:
                res_valid.loc[(res_valid.store == s+1) & (res_valid.item == i+1), 'sales'] = preds_valid
            if pd.Series(preds_valid).hasnans:
                break
            
            # smape for train and valid
            train_smape = round(sMAPE(train.sales.values, preds_train), 4)
            si_scores_df.loc[f's{s+1}_i{i+1}', f'{name}_train'] = train_smape 
            if not future:
                valid_smape = round(sMAPE(valid.sales.values, preds_valid), 4)
                si_scores_df.loc[f's{s+1}_i{i+1}', f'{name}_valid'] = valid_smape 
                print(f'({i+1},{s+1})  train : {train_smape}      valid : {valid_smape}')
            else:
                print(f'({i+1},{s+1})  train : {train_smape}')
            
        item_train = res_train.loc[(res_train.item == i+1)]
        item_valid = res_valid.loc[(res_valid.item == i+1)]
        print('-----------------------------------------------------')
        print(f"Item {i+1} Train sMAPE : {round(sMAPE(item_train.sales, item_train[name]), 4)}")
        if not future:
            print(f"Item {i+1} Valid sMAPE : {round(sMAPE(item_valid.sales, item_valid[name]), 4)}")
        print('-----------------------------------------------------')
        print()

    print()
    overall_train_smape = sMAPE(train_df.sales, res_train[f'{name}'])
    print(f"OVERALL TRAIN sMAPE : {round(overall_train_smape, 4)}")
    scores_df.loc[f'{name}', 'train'] = overall_train_smape

    if not future:
        overall_valid_smape = sMAPE(valid_df.sales, res_valid[f'{name}'])
        print(f"OVERALL VALIDATION sMAPE : {round(overall_valid_smape, 4)}")
        scores_df.loc[f'{name}', 'valid'] = overall_valid_smape
    
    return res_train, res_valid, si_scores_df, scores_df


def plot_smape(si_scores_df, name):
    f, ax = plt.subplots(figsize=(10,3))
    si_scores_df[f'{name}_train'].plot.kde(ax=ax)
    si_scores_df[f'{name}_valid'].plot.kde(ax=ax)
    ax.set_title('sMAPE across all Store-Item Cominations')
    ax.legend()
    plt.show();


def plot_forecast(df, idx, name, ax, r, c, train_valid):
    s = int(idx.split('_')[0][1:])
    i = int(idx.split('_')[1][1:])
    
    actual_ts = df.loc[(df.store==s) & (df.item==i), 'sales']
    preds_ts = df.loc[(df.store==s) & (df.item==i), name]
    score = sMAPE(actual_ts, preds_ts)
    actual_ts.plot(ax=ax[r,c], label='actual', use_index=True)
    preds_ts.plot(ax=ax[r,c], label='preds', use_index=True)
    ax[r,c].set_title(f'{train_valid} - Store {s} Item {i}\nsMAPE: {score}')
    ax[r,c].legend()

def plot_min_max_smape(train_res, valid_res, si_scores_df, name):
    f, ax = plt.subplots(4,2, figsize=(15,12))
    # max smape in train set
    idx_train_max = si_scores_df.idxmax()[f'{name}_train']
    plot_forecast(train_res, idx_train_max, name, ax, 0,0, 'Train Max sMAPE')
    plot_forecast(valid_res, idx_train_max, name, ax, 0,1, 'Valid')
    # min smape in train set
    idx_train_min = si_scores_df.idxmin()[f'{name}_train']
    plot_forecast(train_res, idx_train_min, name, ax, 1,0, 'Train Min sMAPE')
    plot_forecast(valid_res, idx_train_min, name, ax, 1,1, 'Valid')
    # max smape in valid set
    idx_valid_max = si_scores_df.idxmax()[f'{name}_valid']
    plot_forecast(valid_res, idx_valid_max, name, ax, 2,1, 'Valid Max sMAPE')
    plot_forecast(train_res, idx_valid_max, name, ax, 2,0, 'Train')
    # min smape in valid set
    idx_valid_min = si_scores_df.idxmin()[f'{name}_valid']
    plot_forecast(train_res, idx_valid_min, name, ax, 3,0, 'Train')
    plot_forecast(valid_res, idx_valid_min, name, ax, 3,1, 'Valid Min sMAPE')

    plt.tight_layout()
    plt.show();




import torch
import torch.nn as nn
import gc
import warnings
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import os

from utils import set_seed, get_device
from config import config
from data_utils import unzip_and_load_data
from feature_engineering import run_feature_engineering_pipeline
from models import HANTransformer
from training_utils import (
    get_timeseries_kfold_split,
    VolatilityDataset,
    train_model,
    generate_embeddings
)
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
set_seed(42)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPUs available.")

DEVICE = get_device()


train, stock_ids, _ = unzip_and_load_data()
if train is None:
    print("Failed to load data. Exiting.")
    sys.exit(1)


features_path = 'processed_features.parquet'
daily_agg_path = 'processed_daily_agg.parquet'

if os.path.exists(features_path) and os.path.exists(daily_agg_path):
    print("Found pre-computed feature files. Loading them...")
    features_df = pd.read_parquet(features_path)
    daily_agg = pd.read_parquet(daily_agg_path)
    print("Features loaded successfully.")
else:
    print("Pre-computed feature files not found. Running the full feature engineering pipeline...")
    features_df, daily_agg = run_feature_engineering_pipeline(train, stock_ids)
    
    print("Feature engineering complete. Saving results to disk...")
    try:
        features_df.to_parquet(features_path)
        daily_agg.to_parquet(daily_agg_path)
    except Exception as e:
        print(f"Warning: Could not save feature files to disk. Error: {e}")

    print(f"Features saved to '{features_path}' and '{daily_agg_path}'.")


EXCLUDE_COLS = ['stock_id', 'time_id', 'target', 'target_log', 'time_idx', 'pseudo_day', 'tick_size', 'real_price']
SHORT_SCALE_FEATURES = [col for col in features_df.columns if col not in EXCLUDE_COLS]
config.INPUT_DIM_SHORT = len(SHORT_SCALE_FEATURES)

print(f"CRITICAL CONFIG UPDATE: config.INPUT_DIM_SHORT has been set to: {config.INPUT_DIM_SHORT}")
if config.INPUT_DIM_SHORT <= 0:
    raise ValueError("FATAL ERROR: INPUT_DIM_SHORT was not calculated correctly. Check the feature engineering pipeline.")


def main():

    results = {'HAN-Transformer': [], 'Flat-Transformer': [], 'LightGBM': [], 'GARCH(1,1)': [], 'HAN+LGBM Hybrid': []}

    cv_splits = get_timeseries_kfold_split(features_df, n_splits=config.N_SPLITS)

    def get_tabular_features(data_subset, columns_to_get):
        return data_subset[columns_to_get].fillna(0)

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n===== FOLD {fold+1}/{config.N_SPLITS} =====")

        train_fold_df = features_df.iloc[train_idx].copy()
        val_fold_df = features_df.iloc[val_idx].copy()

        scaler = StandardScaler()
        train_fold_df[SHORT_SCALE_FEATURES] = scaler.fit_transform(train_fold_df[SHORT_SCALE_FEATURES])
        val_fold_df[SHORT_SCALE_FEATURES] = scaler.transform(val_fold_df[SHORT_SCALE_FEATURES])

        fold_train_dataset = VolatilityDataset(train_fold_df, daily_agg, config, SHORT_SCALE_FEATURES)
        fold_val_dataset = VolatilityDataset(val_fold_df, daily_agg, config, SHORT_SCALE_FEATURES)
        train_loader_shuffled = DataLoader(fold_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(fold_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        print("\nTraining HAN-Transformer (for feature extraction)...")
        han_model = HANTransformer(config).to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for HAN-Transformer.")
            han_model = nn.DataParallel(han_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

        val_loss_han = train_model(han_model, train_loader_shuffled, val_loader, config, DEVICE)
        results['HAN-Transformer'].append(val_loss_han)
        print(f"Fold {fold+1} Standalone HAN RMSPE: {val_loss_han:.6f}")

        print("\nGenerating embeddings from trained HAN model...")
        train_loader_unshuffled = DataLoader(fold_train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        val_loader_unshuffled = DataLoader(fold_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

        train_embeddings = generate_embeddings(han_model, train_loader_unshuffled, DEVICE)
        val_embeddings = generate_embeddings(han_model, val_loader_unshuffled, DEVICE)
        print(f"Generated train embeddings of shape: {train_embeddings.shape}")
        print(f"Generated validation embeddings of shape: {val_embeddings.shape}")

        print("\nTraining Baseline LightGBM...")
        X_train_lgb = get_tabular_features(train_fold_df, SHORT_SCALE_FEATURES)
        y_train_lgb_log = train_fold_df['target_log']
        y_train_lgb_raw = train_fold_df['target']

        X_val_lgb = get_tabular_features(val_fold_df, SHORT_SCALE_FEATURES)
        y_val_lgb_log = val_fold_df['target_log']
        y_val_lgb_raw = val_fold_df['target'] 

        lgbm_params = {
            'objective':'regression_l1',
            'metric':'rmse',
            'reg_alpha': 5, 'reg_lambda': 5, 'min_data_in_leaf': 1000,
            'max_depth': -1, 'num_leaves': 128, 'colsample_bytree': 0.3, 'learning_rate': 0.02,
            'random_state': 42, 'n_estimators': 8000, 'n_jobs': -1, 'verbose': -1
        }
        lgb_model = lgb.LGBMRegressor(**lgbm_params)
        lgb_model.fit(X_train_lgb, y_train_lgb_log,
                    eval_set=[(X_val_lgb, y_val_lgb_log)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])

        y_pred_lgb_log = lgb_model.predict(X_val_lgb)
        y_pred_lgb_raw = np.exp(y_pred_lgb_log)

        rmspe_lgb = np.sqrt(np.mean(np.square((y_val_lgb_raw - y_pred_lgb_raw) / y_val_lgb_raw)))
        results['LightGBM'].append(rmspe_lgb)
        print(f"Fold {fold+1} Baseline LightGBM RMSPE: {rmspe_lgb:.6f}")

        print("\nTraining HAN+LGBM Hybrid model...")
        X_train_hybrid = np.hstack([X_train_lgb.values, train_embeddings])
        X_val_hybrid = np.hstack([X_val_lgb.values, val_embeddings])

        hybrid_lgb_model = lgb.LGBMRegressor(**lgbm_params)
        hybrid_lgb_model.fit(X_train_hybrid, y_train_lgb_log,
                            eval_set=[(X_val_hybrid, y_val_lgb_log)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])

        y_pred_hybrid_log = hybrid_lgb_model.predict(X_val_hybrid)
        y_pred_hybrid_raw = np.exp(y_pred_hybrid_log)

        rmspe_hybrid = np.sqrt(np.mean(np.square((y_val_lgb_raw - y_pred_hybrid_raw) / y_val_lgb_raw)))
        results['HAN+LGBM Hybrid'].append(rmspe_hybrid)
        print(f"Fold {fold+1} HAN+LGBM Hybrid RMSPE: {rmspe_hybrid:.6f}")

        del han_model, lgb_model, hybrid_lgb_model; gc.collect(); torch.cuda.empty_cache()

        
    results_df = pd.DataFrame(results)
    results_df.loc['mean'] = results_df.mean()
    results_df.loc['std'] = results_df.std()
    print("\n\n===== Final Benchmark Results (RMSPE) =====")
    print(results_df.round(6))

if __name__ == "__main__":
    main()
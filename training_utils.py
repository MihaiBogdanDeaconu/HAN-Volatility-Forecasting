import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast

class VolatilityDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-scale volatility sequences.
    (No changes to this class)
    """
    def __init__(self, data, daily_data, config, short_scale_feature_cols):
        self.data = data.sort_values(by='time_idx').reset_index()
        self.daily_data = daily_data.sort_values(by=['stock_id', 'pseudo_day']).reset_index(drop=True)
        self.config = config
        self.short_scale_features = short_scale_feature_cols

        self.stock_data_map = {stock_id: df.reset_index(drop=True) for stock_id, df in self.data.groupby('stock_id')}
        self.daily_data_map = {stock_id: df.reset_index(drop=True) for stock_id, df in self.daily_data.groupby('stock_id')}

        self.index_mapper = self.data[['stock_id']].copy()
        self.index_mapper['intra_stock_idx'] = self.data.groupby('stock_id').cumcount()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_info = self.index_mapper.iloc[idx]
        stock_id = row_info['stock_id']
        intra_stock_idx = row_info['intra_stock_idx']

        row = self.stock_data_map[stock_id].iloc[intra_stock_idx]
        current_day = row['pseudo_day']

        stock_df = self.stock_data_map[stock_id]
        start_loc = max(0, intra_stock_idx - self.config.SEQ_LEN_SHORT + 1)
        short_seq_df = stock_df.iloc[start_loc:intra_stock_idx + 1]
        short_features = short_seq_df[self.short_scale_features].values

        if len(short_features) < self.config.SEQ_LEN_SHORT:
            padding = np.zeros((self.config.SEQ_LEN_SHORT - len(short_features), self.config.INPUT_DIM_SHORT))
            short_features = np.vstack([padding, short_features])

        daily_stock_df = self.daily_data_map.get(stock_id)
        if daily_stock_df is None or daily_stock_df.empty:
            mid_features = np.zeros(self.config.SEQ_LEN_MID)
            long_features = np.zeros(self.config.SEQ_LEN_LONG)
        else:
            day_pos = daily_stock_df[daily_stock_df['pseudo_day'] < current_day].index.max()
            if pd.isna(day_pos): day_pos = -1

            mid_start_pos = max(0, day_pos - self.config.SEQ_LEN_MID + 1)
            mid_features = daily_stock_df.iloc[mid_start_pos : day_pos + 1]['daily_vol'].values
            if len(mid_features) < self.config.SEQ_LEN_MID:
                padding = np.zeros(self.config.SEQ_LEN_MID - len(mid_features))
                mid_features = np.concatenate([padding, mid_features])

            long_start_pos = max(0, day_pos - self.config.SEQ_LEN_LONG + 1)
            long_features = daily_stock_df.iloc[long_start_pos : day_pos + 1]['weekly_vol'].values
            if len(long_features) < self.config.SEQ_LEN_LONG:
                padding = np.zeros(self.config.SEQ_LEN_LONG - len(long_features))
                long_features = np.concatenate([padding, long_features])

        return {
            'x_short': torch.tensor(short_features, dtype=torch.float32).nan_to_num(),
            'x_mid': torch.tensor(mid_features, dtype=torch.float32).unsqueeze(-1).nan_to_num(),
            'x_long': torch.tensor(long_features, dtype=torch.float32).unsqueeze(-1).nan_to_num(),
            'y_raw': torch.tensor(row['target'], dtype=torch.float32)
        }

def generate_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            x_short, x_mid, x_long = (batch['x_short'].to(device), batch['x_mid'].to(device), batch['x_long'].to(device))
            embeddings = model(x_short, x_mid, x_long, return_embedding=True)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def rmspe_loss(y_pred_log, y_true_raw):
    y_pred_raw = torch.exp(y_pred_log.squeeze())
    return torch.sqrt(torch.mean(((y_true_raw.squeeze() - y_pred_raw) / (y_true_raw.squeeze() + 1e-8)) ** 2))

def get_timeseries_kfold_split(data, n_splits=5):
    time_ids = np.sort(data['time_id'].unique())
    kf = KFold(n_splits=n_splits, shuffle=False)
    splits = []
    for train_time_idx, val_time_idx in kf.split(time_ids):
        train_times = time_ids[train_time_idx]
        val_times = time_ids[val_time_idx]
        train_indices = data[data['time_id'].isin(train_times)].index.values
        val_indices = data[data['time_id'].isin(val_times)].index.values
        splits.append((train_indices, val_indices))
        print(f"Fold created: Train size={len(train_indices)}, Val size={len(val_indices)}")
    return splits

def train_model(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    warmup_steps = config.WARMUP_EPOCHS * len(train_loader)
    total_steps = config.EPOCHS * len(train_loader)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    def warmup_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    scaler = GradScaler()
    best_val_rmspe = float('inf')
    clip_grad_norm = 1.0

    print(f"Starting training for {config.EPOCHS} epochs with a single cosine cycle schedule...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            x_short, x_mid, x_long, y_raw_true = (
                batch['x_short'].to(device), batch['x_mid'].to(device),
                batch['x_long'].to(device), batch['y_raw'].to(device)
            )
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                y_pred_log = model(x_short, x_mid, x_long)
                loss = rmspe_loss(y_pred_log, y_raw_true)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()


            current_step_num = epoch * len(train_loader) + step
            if current_step_num < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_rmspe = 0
        with torch.no_grad():
            for batch in val_loader:
                x_short, x_mid, x_long, y_raw_true = (
                    batch['x_short'].to(device), batch['x_mid'].to(device),
                    batch['x_long'].to(device), batch['y_raw'].to(device)
                )
                with autocast():
                    y_pred_log = model(x_short, x_mid, x_long)
                    rmspe = rmspe_loss(y_pred_log, y_raw_true)
                total_val_rmspe += rmspe.item()
        avg_val_rmspe = total_val_rmspe / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val RMSPE: {avg_val_rmspe:.6f}, LR: {current_lr:.6e}")

        if avg_val_rmspe < best_val_rmspe:
            best_val_rmspe = avg_val_rmspe
            print(f"    New best validation RMSPE: {best_val_rmspe:.6f}")
            # torch.save(model.state_dict(), 'best_model.pth')

    return best_val_rmspe
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import LatentDirichletAllocation

from data_utils import load_book_data, load_trade_data


def calc_wap1(df: pd.DataFrame) -> pd.Series:
    return (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

def calc_wap2(df: pd.DataFrame) -> pd.Series:
    return (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def log_return(series: np.ndarray):
    return np.log(series).diff()

def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['time_id', 'stock_id']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + [str(i) for i in c]))
    return ret

def make_book_feature(stock_id, block='train'):
    book = load_book_data(stock_id)
    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)
    book['log_return1'] = book.groupby(['time_id'])['wap1'].transform(log_return)
    book['log_return2'] = book.groupby(['time_id'])['wap2'].transform(log_return)
    book['log_return_ask1'] = book.groupby(['time_id'])['ask_price1'].transform(log_return)
    book['log_return_ask2'] = book.groupby(['time_id'])['ask_price2'].transform(log_return)
    book['log_return_bid1'] = book.groupby(['time_id'])['bid_price1'].transform(log_return)
    book['log_return_bid2'] = book.groupby(['time_id'])['bid_price2'].transform(log_return)
    book['wap_balance'] = abs(book['wap1'] - book['wap2'])
    book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / ((book['ask_price1'] + book['bid_price1']) / 2)
    book['bid_spread'] = book['bid_price1'] - book['bid_price2']
    book['ask_spread'] = book['ask_price1'] - book['ask_price2']
    book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])
    book['volume_imbalance'] = abs((book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2']))

    features = {
        'seconds_in_bucket':['count'], 'wap1':[np.sum, np.mean, np.std], 'wap2':[np.sum, np.mean, np.std],
        'log_return1':[np.sum, realized_volatility, np.mean, np.std], 'log_return2':[np.sum, realized_volatility, np.mean, np.std],
        'log_return_ask1':[np.sum, realized_volatility, np.mean, np.std], 'log_return_ask2':[np.sum, realized_volatility, np.mean, np.std],
        'log_return_bid1':[np.sum, realized_volatility, np.mean, np.std], 'log_return_bid2':[np.sum, realized_volatility, np.mean, np.std],
        'wap_balance':[np.sum, np.mean, np.std], 'price_spread':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std], 'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std], 'volume_imbalance':[np.sum, np.mean, np.std]
    }
    agg = book.groupby('time_id').agg(features).reset_index(drop=False)
    agg.columns = flatten_name('book', agg.columns)
    agg['stock_id'] = stock_id
    for time in [450, 300, 150]:
        d = book[book['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'book_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')
    return agg

def make_trade_feature(stock_id, block='train'):
    trade = load_trade_data(stock_id)
    trade['log_return'] = trade.groupby('time_id')['price'].transform(log_return)
    features = {
        'log_return':[realized_volatility], 'seconds_in_bucket':['count'],
        'size':[np.sum], 'order_count':[np.mean],
    }
    agg = trade.groupby('time_id').agg(features).reset_index()
    agg.columns = flatten_name('trade', agg.columns)
    agg['stock_id'] = stock_id
    for time in [450, 300, 150]:
        d = trade[trade['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'trade_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')
    return agg

def make_book_feature_v2(stock_id, block='train'):
    book = load_book_data(stock_id)
    prices = book.set_index('time_id')[['bid_price1', 'ask_price1', 'bid_price2', 'ask_price2']]
    ticks = {}
    for tid, group in prices.groupby(level='time_id'):
        price_list = group.values.flatten()
        price_diff = sorted(np.diff(sorted(set(price_list))))
        if len(price_diff) > 0: ticks[tid] = price_diff[0]
        else: ticks[tid] = np.nan
    dst = pd.DataFrame.from_dict(ticks, orient='index', columns=['tick_size'])
    dst['stock_id'] = stock_id
    return dst.reset_index().rename(columns={'index':'time_id'})

class Neighbors:
    def __init__(self, name: str, pivot: pd.DataFrame, p: float, metric: str = 'minkowski', metric_params: dict = None, exclude_self: bool = False):
        self.name, self.exclude_self, self.p, self.metric = name, exclude_self, p, metric
        nn = NearestNeighbors(n_neighbors=80, p=p, metric=metric, metric_params=metric_params, n_jobs=-1)
        nn.fit(pivot)
        _, self.neighbors = nn.kneighbors(pivot, return_distance=True)
        self.columns = self.index = self.feature_values = self.feature_col = None
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str): raise NotImplementedError()
    def make_nn_feature(self, n=5, agg=np.mean) -> pd.DataFrame:
        start = 1 if self.exclude_self else 0
        pivot_aggs = pd.DataFrame(agg(self.feature_values[start:n, :, :], axis=0), columns=self.columns, index=self.index)
        dst = pivot_aggs.unstack().reset_index()
        dst.columns = ['stock_id', 'time_id', f'{self.feature_col}_nn{n}_{self.name}_{agg.__name__}']
        return dst

class TimeIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str):
        feature_pivot = df.pivot(index='time_id', columns='stock_id', values=feature_col).fillna(df[feature_col].mean())
        feature_values = np.zeros((80, *feature_pivot.shape))
        for i in range(80): feature_values[i, :, :] += feature_pivot.values[self.neighbors[:, i], :]
        self.columns, self.index, self.feature_values, self.feature_col = list(feature_pivot.columns), list(feature_pivot.index), feature_values, feature_col

class StockIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str):
        feature_pivot = df.pivot(index='time_id', columns='stock_id', values=feature_col).fillna(df[feature_col].mean())
        feature_values = np.zeros((80, *feature_pivot.shape))
        for i in range(80): feature_values[i, :, :] += feature_pivot.values[:, self.neighbors[:, i]]
        self.columns, self.index, self.feature_values, self.feature_col = list(feature_pivot.columns), list(feature_pivot.index), feature_values, feature_col

def make_nearest_neighbor_feature(df: pd.DataFrame, time_id_neighbors, stock_id_neighbors) -> pd.DataFrame:
    df2 = df.copy()
    feature_cols_stock = {'book.log_return1.realized_volatility':[np.mean,np.min,np.max,np.std], 'trade.seconds_in_bucket.count':[np.mean], 'trade.tau':[np.mean], 'trade_150.tau':[np.mean], 'book.tau':[np.mean], 'trade.size.sum':[np.mean], 'book.seconds_in_bucket.count':[np.mean]}
    feature_cols_time = {'book.log_return1.realized_volatility':[np.mean,np.min,np.max,np.std], 'real_price':[np.max,np.mean,np.min], 'trade.seconds_in_bucket.count':[np.mean], 'trade.tau':[np.mean], 'trade.size.sum':[np.mean], 'book.seconds_in_bucket.count':[np.mean]}

    for feature_col, aggs in feature_cols_stock.items():
        if feature_col not in df2.columns: continue
        for nn in stock_id_neighbors:
            nn.rearrange_feature_values(df2, feature_col)
            for agg in aggs:
                for n in [10, 20, 40]:
                    dst = nn.make_nn_feature(n, agg)
                    df2 = pd.merge(df2, dst, on=['time_id', 'stock_id'], how='left')

    for feature_col, aggs in feature_cols_time.items():
        if feature_col not in df2.columns: continue
        for nn in time_id_neighbors:
            nn.rearrange_feature_values(df2, feature_col)
            for agg in aggs:
                for n in [3, 5, 10, 20, 40]:
                    dst = nn.make_nn_feature(n, agg)
                    df2 = pd.merge(df2, dst, on=['time_id', 'stock_id'], how='left')

    for sz in [3, 5, 10, 20, 40]:
        for T in ['time_price_c', 'time_price_m', 'time_vol_l1', 'time_size_m', 'time_size_c']:
            for agg_type in ['mean', 'amin', 'amax']:
                denominator_rp = f"real_price_nn{sz}_{T}_{agg_type}"
                denominator_vol = f"book.log_return1.realized_volatility_nn{sz}_{T}_{agg_type}"
                if denominator_rp in df2.columns: df2[f'real_price_rank_{sz}_{T}_{agg_type}'] = df2['real_price'] / df2[denominator_rp]
                if denominator_vol in df2.columns: df2[f'vol_rank_{sz}_{T}_{agg_type}'] = df2['book.log_return1.realized_volatility'] / df2[denominator_vol]
    return df2
def run_feature_engineering_pipeline(train, stock_ids):

    print("Step 1: Generating rich base features...")
    books = Parallel(n_jobs=-1)(delayed(make_book_feature)(i, "train") for i in stock_ids)
    trades = Parallel(n_jobs=-1)(delayed(make_trade_feature)(i, "train") for i in stock_ids)
    df_base = pd.merge(pd.concat(books), pd.concat(trades), on=['stock_id', 'time_id'], how='left')

    print("Step 2: Generating tick size features...")
    books_v2 = Parallel(n_jobs=-1)(delayed(make_book_feature_v2)(i, "train") for i in stock_ids)
    features_df = pd.merge(train, df_base, on=['stock_id', 'time_id'], how='left')
    features_df = pd.merge(features_df, pd.concat(books_v2), on=['stock_id', 'time_id'], how='left')

    print("Step 3: Generating full KNN features...")
    features_df['real_price'] = 0.01 / features_df['tick_size']
    df_pv = features_df[['stock_id','time_id']].copy()

    features_df['trade.tau'] = np.sqrt(1 / features_df['trade.seconds_in_bucket.count'])
    features_df['trade_150.tau'] = np.sqrt(1 / features_df['trade_150.seconds_in_bucket.count'])
    features_df['book.tau'] = np.sqrt(1 / features_df['book.seconds_in_bucket.count'])
    df_pv['price'] = 0.01 / features_df['tick_size']
    df_pv['vol'] = features_df['book.log_return1.realized_volatility']
    df_pv['trade.size.sum'] = features_df['book.total_volume.sum']

    pivot_price = pd.DataFrame(minmax_scale(df_pv.pivot(index='time_id', columns='stock_id', values='price').fillna(df_pv.price.mean())))
    pivot_vol = pd.DataFrame(minmax_scale(df_pv.pivot(index='time_id', columns='stock_id', values='vol').fillna(df_pv.vol.mean())))
    pivot_size = pd.DataFrame(minmax_scale(df_pv.pivot(index='time_id', columns='stock_id', values='trade.size.sum').fillna(df_pv['trade.size.sum'].mean())))

    time_id_neighbors = [
        TimeIdNeighbors('time_price_c', pivot_price, p=2, metric='canberra', exclude_self=True),
        TimeIdNeighbors('time_price_m', pivot_price, p=2, metric='mahalanobis', metric_params={'VI': np.linalg.pinv(np.cov(pivot_price.values.T))}),
        TimeIdNeighbors('time_vol_l1', pivot_vol, p=1),
        TimeIdNeighbors('time_size_m', pivot_size, p=2, metric='mahalanobis', metric_params={'VI': np.linalg.pinv(np.cov(pivot_size.values.T))}),
        TimeIdNeighbors('time_size_c', pivot_size, p=2, metric='canberra')
    ]
    stock_id_neighbors = [
        StockIdNeighbors('stock_price_l1', minmax_scale(pivot_price.transpose()), p=1, exclude_self=True),
        StockIdNeighbors('stock_vol_l1', minmax_scale(pivot_vol.transpose()), p=1, exclude_self=True)
    ]

    features_df = make_nearest_neighbor_feature(features_df, time_id_neighbors, stock_id_neighbors)

    print("Step 4: Final processing and all derived features...")

    rank_cols = [c for c in features_df.columns if 'order_count.mean' in c or 'total_volume.sum' in c or 'total_volume.mean' in c or 'total_volume.std' in c]
    for col in rank_cols:
        features_df[col] = features_df.groupby('time_id')[col].rank()

    features_df = features_df.sort_values(by=['stock_id', 'book.total_volume.sum']).reset_index(drop=True)
    features_df['vol_roll_by_vol'] = features_df.groupby('stock_id')['book.log_return1.realized_volatility'].rolling(window=10, min_periods=1).mean().reset_index(0,drop=True)
    features_df = features_df.sort_values(by=['stock_id', 'time_id']).reset_index(drop=True)

    log_cols = [c for c in features_df.columns if 'size.sum' in c or 'volume_imbalance' in c]
    for col in log_cols:
        features_df[col] = np.log(features_df[col] + 1)

    lda = LatentDirichletAllocation(n_components=3, random_state=0)
    stock_id_emb = pd.DataFrame(lda.fit_transform(pivot_vol.transpose()), index=df_pv.pivot(index='time_id', columns='stock_id', values='vol').columns)
    for i in range(3):
        features_df[f'stock_id_emb{i}'] = features_df['stock_id'].map(stock_id_emb[i])

    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Feature pipeline complete.")

    print("\nRecreating mid and long-scale aggregates for sequence models...")
    time_ids_sorted = sorted(features_df['time_id'].unique())
    time_id_map = {tid: i for i, tid in enumerate(time_ids_sorted)}
    features_df['time_idx'] = features_df['time_id'].map(time_id_map)

    features_df.fillna(0, inplace=True)

    num_days_approx = len(time_ids_sorted) / (428932 / 3830)
    features_df['pseudo_day'] = (features_df['time_idx'] / (len(time_ids_sorted) / num_days_approx)).astype(int)
    daily_agg = features_df.groupby(['stock_id', 'pseudo_day']).agg(daily_vol=('target', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
    daily_agg = daily_agg.sort_values(by=['stock_id', 'pseudo_day']).reset_index(drop=True)
    daily_agg['weekly_vol'] = daily_agg.groupby('stock_id')['daily_vol'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    print("Aggregates ready.")

    print(f"\nFinal total number of features: {features_df.shape[1]}")

    features_df['target_log'] = np.log(features_df['target'] + 1e-8)

    print("Created 'target_log' for model training.")
    return features_df, daily_agg
from finlab import ml
from finlab.dataframe import FinlabDataFrame

import os
import abc
import yaml
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging as logger
from pathlib import Path
from typing import Iterable, List, Union
from functools import partial
from concurrent.futures import ProcessPoolExecutor


import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.loader import DataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158, Alpha360
from qlib.utils import fname_to_code, code_to_fname, init_instance_by_config


class DumpDataBase:
    """
    Base class for dumping data to Qlib format.

    Args:
        csv_path (str): The path to the CSV file or directory containing the CSV files.
        qlib_dir (str): The directory where the Qlib data will be saved.
        backup_dir (str, optional): The directory where the backup of the Qlib data will be saved. Defaults to None.
        freq (str, optional): The frequency of the data. Defaults to "day".
        max_workers (int, optional): The maximum number of workers for parallel processing. Defaults to 16.
        date_field_name (str, optional): The name of the date field in the CSV file. Defaults to "date".
        file_suffix (str, optional): The suffix of the CSV file. Defaults to ".csv".
        symbol_field_name (str, optional): The name of the symbol field in the CSV file. Defaults to "symbol".
        exclude_fields (str, optional): The fields to exclude from the dumped data. Defaults to "".
        include_fields (str, optional): The fields to include in the dumped data. Defaults to "".
        limit_nums (int, optional): The maximum number of CSV files to process. Defaults to None.
    """

    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        csv_path = Path(csv_path).expanduser()
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _format_datetime(self, datetime_d):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df, *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        df[self.date_field_name] = df[self.date_field_name].astype(str).astype(np.datetime64)
        # df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (self._include_fields)

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = list(map(lambda x: self._format_datetime(x), calendars_data))
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype('datetime64[ns]')
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        # used when creating a bin file
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data, calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return

        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()

class DumpDataAll(DumpDataBase):
    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.csv_files, executor.map(_fun, self.csv_files)
                ):
                    all_datetime = all_datetime | _set_calendars
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        _begin_time = self._format_datetime(_begin_time)
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")

    def _dump_features(self):
        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()

def get_region(market):
    return ml.get_market().__class__.__name__.replace('MarketInfo', '').lower()

def dump(freq='day'):
    """產生Qlib 於台股的資料庫
    Examples:
        ```py
        import qlib
        import finlab.ml.qlib as q

        q.dump() # generate tw stock database
        q.init() # initiate tw stock to perform machine leraning tasks (similar to qlib.init)

        import qlib
        # qlib functions and operations
        ```
    """
    
    market = ml.get_market()
    region = get_region(market)

    csv_path = f'~/.qlib/csv_data/{region}_data'
    qlib_dir = f'~/.qlib/qlib_data/{region}_data'
    include_fields = "open,close,high,low,volume,factor"

    if not Path(csv_path).expanduser().exists():
        Path(csv_path).expanduser().mkdir(parents=True)
    if not Path(qlib_dir).expanduser().exists():
        Path(qlib_dir).expanduser().mkdir(parents=True)

    c = market.get_price('close', adj=False)
    ac = market.get_price('close', adj=True)
    o = market.get_price('open', adj=False)
    h = market.get_price('high', adj=False)
    l = market.get_price('low', adj=False)
    v = market.get_price('volume', adj=False)

    assert c is not None
    assert ac is not None
    assert o is not None
    assert h is not None
    assert l is not None
    assert v is not None

    for s in c.columns:
        pd.DataFrame({
            'date':c.index.values,
            'volume': v[s].values,
            'high': h[s].values,
            'low': l[s].values,
            'close': c[s].values,
            'open': o[s].values,
            'factor': ac[s].values / c[s].values,
            'symbol': s
            }).to_csv(Path(csv_path).expanduser() / f"{s}.csv")

    dumper = DumpDataAll(csv_path, qlib_dir, include_fields=include_fields, freq=freq)
    dumper()


qlib_initialized = False

def init():
    """Qlib 初始化 (類似於台股版 qlib.init() 但更簡單易用)
    Examples:
        ```py
        import qlib
        import finlab.ml.qlib as q

        q.dump() # generate tw stock database
        q.init() # initiate tw stock to perform machine leraning tasks (similar to qlib.init)

        import qlib
        # qlib functions and operations
        ```
    """
    region = get_region(ml.get_market())
    try:
        from qlib import config
        config._default_region_config[region] = \
                dict(trade_unit=1000, limit_threshold=0.1, deal_price='close')
    except:
        pass

    global qlib_initialized

    if not qlib_initialized:
        qlib.init(provider_uri=f'~/.qlib/qlib_data/{region}_data', 
                  region=region)
        qlib_initialized = True

def alpha(handler='Alpha158', **kwargs):

    """產生 Qlib 的特徵
    Args:
        handler (str): 預設為 'alpha158' 也可以設定成 'Alpha360'
    Examples:
        ```py
        import finlab.ml.qlib as q
        features = q.alpha('Alpha158')
        ```
    """
    init()

    if handler == 'Alpha158':
        h = Alpha158(instruments=D.instruments(market='all'), **kwargs)
    elif handler == 'Alpha360':
        h = Alpha360(instruments=D.instruments(market='all'), **kwargs)
    else:
        raise Exception(f"Handler {handler} not supported.")

    alpha = h.fetch(col_set="feature")
    return alpha



class CustomDataLoader(DataLoader):
    def __init__(self, d):

        from finlab.utils import get_tmp_dir
        tmp_dir = get_tmp_dir()
        self.data_path = os.path.join(tmp_dir, 'dataset.pickle')
        d.to_pickle(self.data_path)
        del d
        
    def load(self, instruments, start_time=None, end_time=None) -> pd.DataFrame:
        d = pd.read_pickle(self.data_path)
        t = d.index.get_level_values('datetime')

        selected = t.notna()
        if start_time:
            selected &= t > start_time
        if end_time:
            selected &= t < end_time
        if instruments:
            ins = d.index.get_level_values('instrument')
            selected &= ins.isin(instruments) 

        return d.loc[selected]


def make_datasetH(X, y=None, _DEFAULT_LEARN_PROCESSORS=[
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        {"class": "DropnaLabel"},
    ], train_test_split=(0.7, 0.15, 0.15), segments=None):

    is_train = y is not None

    if is_train:

        if segments is None:
            tmin = X.index.get_level_values('datetime').min()
            tmax = X.index.get_level_values('datetime').max()
            tsplit1 = (tmax - tmin) * train_test_split[0] + tmin
            tsplit2 = (tmax - tmin) * (train_test_split[0] + train_test_split[1]) + tmin
            segments = dict(
                train=(tmin, tsplit1),
                valid=(tsplit1 + datetime.timedelta(days=1), tsplit2),
                test=(tsplit2 + datetime.timedelta(days=1), tmax)
            )

        d = pd.concat([
            X, 
            y.replace([np.inf, -np.inf], np.nan).to_frame(name='LABEL0')],
            axis=1, 
            keys=['feature', 'label']
            ).sort_index()

        dl = CustomDataLoader(d)        
        dh = DatasetH(handler=DataHandlerLP(data_loader=dl, learn_processors=_DEFAULT_LEARN_PROCESSORS), segments=segments)
        return dh

    x = X.copy()
    x.columns = pd.MultiIndex.from_tuples([('feature', x) for x in x.columns])

    dl = CustomDataLoader(x)

    tmin = x.index.get_level_values('datetime').min()
    tmax = x.index.get_level_values('datetime').max()

    segments = {
        'test': (tmin, tmax)
    }
    
    return DatasetH(handler=DataHandlerLP(data_loader=dl), segments=segments)


class WrapperModel():

    def __init__(self, model_config):
        init()
        self.config = model_config
        self.model = None
        
    def fit(self, X_train, y_train, segments=None, **fit_params):

        config = self.config

        if 'kwargs' in config and 'd_feat' in config['kwargs']:
            config['kwargs']['d_feat'] = X_train.shape[1]

        if 'kwargs' in config and 'pt_model_kwargs' in config['kwargs'] and 'input_dim' in config['kwargs']['pt_model_kwargs']:
            config['kwargs']['pt_model_kwargs']['input_dim'] = X_train.shape[1]

        self.model = init_instance_by_config(self.config)
        dh = make_datasetH(X_train, y_train, segments=segments)
        self.model.fit(dh, **fit_params)

    def predict(self, X_test):
        
        dh = make_datasetH(X_test, None)
        return FinlabDataFrame(pd.Series(self.model.predict(dh), index=X_test.index)
            .reset_index()
            .pivot(index='datetime', columns='instrument', values=0))


def LGBModel():
    """LGBModel is a wrapper model for LightGBM model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.LGBModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return  WrapperModel(yaml.safe_load("""
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
        loss: mse
        colsample_bytree: 0.8879
        learning_rate: 0.2
        subsample: 0.8789
        lambda_l1: 205.6999
        lambda_l2: 580.9768
        max_depth: 8
        num_leaves: 210
        num_threads: 20
    """))

def XGBModel():
    """
    XGBModel is a wrapper model for XGBoost model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.XGBModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: XGBModel
module_path: qlib.contrib.model.xgboost
kwargs:
    eval_metric: rmse
    colsample_bytree: 0.8879
    eta: 0.0421
    max_depth: 8
    n_estimators: 647
    subsample: 0.8789
    nthread: 20
"""))

def DEnsmbleModel():
    """
    DEnsmbleModel is a wrapper model for Double Ensemble model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.DEnsmbleModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: DEnsembleModel
module_path: qlib.contrib.model.double_ensemble
kwargs:
    base_model: "gbm"
    loss: mse
    num_models: 3
    enable_sr: True
    enable_fs: True
    alpha1: 1
    alpha2: 1
    bins_sr: 10
    bins_fs: 5
    decay: 0.5
    sample_ratios:
        - 0.8
        - 0.7
        - 0.6
        - 0.5
        - 0.4
    sub_weights:
        - 1
        - 1
        - 1
    epochs: 28
    colsample_bytree: 0.8879
    learning_rate: 0.0421
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
    verbosity: -1
"""))

def CatBoostModel():
    """
    CatBoostModel is a wrapper model for CatBoost model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.CatBoostModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: CatBoostModel
module_path: qlib.contrib.model.catboost_model
kwargs:
    loss: RMSE
    learning_rate: 0.0421
    subsample: 0.8789
    max_depth: 6
    num_leaves: 100
    thread_count: 20
    grow_policy: Lossguide
"""))

def LinearModel():
    """
    LinearModel is a wrapper model for Linear model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.LinearModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: LinearModel
module_path: qlib.contrib.model.linear
kwargs:
    estimator: ols
"""))

def TabnetModel():
    """
    TabnetModel is a wrapper model for Tabnet model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.TabnetModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: TabnetModel
module_path: qlib.contrib.model.pytorch_tabnet
kwargs:
    d_feat: 8
    pretrain: False
    seed: 993
"""))

def DNNModel():
    """
    DNNModel is a wrapper model for Deep Neural Network model.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.DNNModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: DNNModelPytorch
module_path: qlib.contrib.model.pytorch_nn
kwargs:
    loss: mse
    lr: 0.002
    optimizer: adam
    max_steps: 8000
    batch_size: 8192
    GPU: 0
    weight_decay: 0.0002
    pt_model_kwargs:
      input_dim: 8
"""))

def SFMModel():
    """
    SFMModel is a wrapper model for SFM.
    ```py
    import finlab.ml.qlib as q

    # build X_train, y_train, X_test

    model = q.SFMModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```
    """
    return WrapperModel(yaml.safe_load("""
class: SFM
module_path: qlib.contrib.model.pytorch_sfm
kwargs:
    d_feat: 6
    hidden_size: 64
    output_dim: 32
    freq_dim: 25
    dropout_W: 0.5
    dropout_U: 0.5
    n_epochs: 20
    lr: 0.001
    batch_size: 1600
    early_stop: 20
    eval_steps: 5
    loss: mse
    optimizer: adam
    GPU: 0
"""))

def get_models():
    """Return a list of available models.
    Examples:
        ```py
        import finlab.ml.qlib as q

        models = q.get_models()
        print(models)
        ```
        output:

        { 'LGBModel': LGBModel, 'XGBModel': XGBModel, 'DEnsmbleModel': DEnsmbleModel, 'CatBoostModel': CatBoostModel, 'LinearModel': LinearModel, 'TabnetModel': TabnetModel, 'DNNModel': DNNModel, 'SFMModel': SFMModel}

    """
    return {
        'LGBModel': LGBModel,
        'XGBModel': XGBModel,
        'DEnsmbleModel': DEnsmbleModel,
        'CatBoostModel': CatBoostModel,
        'LinearModel': LinearModel,
        'TabnetModel': TabnetModel,
        'DNNModel': DNNModel,
    }
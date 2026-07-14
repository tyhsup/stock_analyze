import requests
import urllib3
import pandas as pd
import json
import sys
from datetime import datetime
from django.core.management.base import BaseCommand
from market_data.models import MacroUS, MacroTW

# 禁用不安全請求警告 (因應央行 API SSL 憑證問題)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Command(BaseCommand):
    help = 'Fetch and update macroeconomic data for US (FRED) and Taiwan (Central Bank)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--country',
            type=str,
            help='Country to update: "us" or "tw". If not specified, updates both.'
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output result summary in JSON format to stdout for Agent integration.'
        )

    def handle(self, *args, **options):
        country = options.get('country')
        output_json = options.get('json')

        if output_json:
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                pass

        status_report = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "updates": {}
        }

        try:
            if not country or country.lower() == 'us':
                us_count = self._fetch_us_data()
                status_report["updates"]["us"] = {"status": "success", "records_updated": us_count}

            if not country or country.lower() == 'tw':
                tw_count = self._fetch_tw_data()
                status_report["updates"]["tw"] = {"status": "success", "records_updated": tw_count}

            if output_json:
                print(json.dumps(status_report, ensure_ascii=False))
            else:
                self.stdout.write(self.style.SUCCESS(f"Macroeconomic data update completed. US: {status_report['updates'].get('us', {}).get('records_updated', 0)} updated. TW: {status_report['updates'].get('tw', {}).get('records_updated', 0)} updated."))

        except Exception as e:
            error_report = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            if output_json:
                print(json.dumps(error_report, ensure_ascii=False))
                sys.exit(1)
            else:
                self.stderr.write(self.style.ERROR(f"Macro update failed: {e}"))
                raise e

    def _fetch_us_data(self) -> int:
        """
        從 FRED 下載美國總經數據並更新資料庫
        """
        metrics_urls = {
            'FEDFUNDS': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS', # 聯邦基金有效利率
            'CPIAUCSL': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL', # CPI 指數
            'UNRATE': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE',     # 失業率
            'PAYEMS': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=PAYEMS',      # 非農就業人口
            'M1SL': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=M1SL',          # M1 貨幣供給
            'M2SL': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL',          # M2 貨幣供給
            'CPILFESL': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPILFESL',            # 新增美國指標
            'PCE': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCE',            # 個人消費支出
            'GDPC1': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1',        # 實質國內生產毛額 GDP
            'PSAVERT': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=PSAVERT',    # 個人儲蓄率
            'PI': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=PI',              # 個人收入
            'MCOILWTICO': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILWTICO' # 國際原油價格 WTI
        }

        # 限定 15 年時間跨度
        cutoff_date = datetime(2011, 1, 1)

        data_dict = {}
        for metric, url in metrics_urls.items():
            try:
                df = pd.read_csv(url)
                df['DATE'] = pd.to_datetime(df['observation_date'])
                df['VALUE'] = pd.to_numeric(df[metric], errors='coerce')
                df.dropna(subset=['VALUE'], inplace=True)
                # 過濾最近 15 年
                df = df[df['DATE'] >= cutoff_date]
                data_dict[metric] = df.sort_values('DATE').reset_index(drop=True)
            except Exception as e:
                # 【防禦性編碼】：抓取失敗時僅拋出警告，不中斷主更新流程，確保其他指標更新不受影響
                print(f"[WARNING] Failed to fetch US metric {metric} from FRED: {e}")
                data_dict[metric] = pd.DataFrame()

        # 1. 美國 CPI YoY 年增率計算
        cpi_df = pd.DataFrame()
        if 'CPIAUCSL' in data_dict and not data_dict['CPIAUCSL'].empty:
            cpi_df = data_dict['CPIAUCSL'].copy()
            cpi_df['YoY'] = (cpi_df['VALUE'] - cpi_df['VALUE'].shift(12)) / cpi_df['VALUE'].shift(12) * 100
            cpi_df.dropna(subset=['YoY'], inplace=True)

        # 2. 非農就業人口按月新增變動 (NFP Change) 計算
        payems_df = pd.DataFrame()
        if 'PAYEMS' in data_dict and not data_dict['PAYEMS'].empty:
            payems_df = data_dict['PAYEMS'].copy()
            payems_df['NFP_Change'] = (payems_df['VALUE'] - payems_df['VALUE'].shift(1)) * 1000  # 轉為人數
            payems_df.dropna(subset=['NFP_Change'], inplace=True)

        # 3. 美國 M1 YoY 年增率計算
        m1_df = pd.DataFrame()
        if 'M1SL' in data_dict and not data_dict['M1SL'].empty:
            m1_df = data_dict['M1SL'].copy()
            m1_df['YoY'] = (m1_df['VALUE'] - m1_df['VALUE'].shift(12)) / m1_df['VALUE'].shift(12) * 100
            m1_df.dropna(subset=['YoY'], inplace=True)

        # 4. 美國 M2 YoY 年增率計算
        m2_df = pd.DataFrame()
        if 'M2SL' in data_dict and not data_dict['M2SL'].empty:
            m2_df = data_dict['M2SL'].copy()
            m2_df['YoY'] = (m2_df['VALUE'] - m2_df['VALUE'].shift(12)) / m2_df['VALUE'].shift(12) * 100
            m2_df.dropna(subset=['YoY'], inplace=True)

        # 5. 美國核心 CPI YoY 年增率計算
        core_cpi_df = pd.DataFrame()
        if 'CPILFESL' in data_dict and not data_dict['CPILFESL'].empty:
            core_cpi_df = data_dict['CPILFESL'].copy()
            core_cpi_df['YoY'] = (core_cpi_df['VALUE'] - core_cpi_df['VALUE'].shift(12)) / core_cpi_df['VALUE'].shift(12) * 100
            core_cpi_df.dropna(subset=['YoY'], inplace=True)

        # 6. 美國 PCE YoY 年增率計算
        pce_df = pd.DataFrame()
        if 'PCE' in data_dict and not data_dict['PCE'].empty:
            pce_df = data_dict['PCE'].copy()
            pce_df['YoY'] = (pce_df['VALUE'] - pce_df['VALUE'].shift(12)) / pce_df['VALUE'].shift(12) * 100
            pce_df.dropna(subset=['YoY'], inplace=True)

        # 7. 美國 Real GDP YoY 年增率計算 (季資料)
        gdp_df = pd.DataFrame()
        if 'GDPC1' in data_dict and not data_dict['GDPC1'].empty:
            gdp_df = data_dict['GDPC1'].copy()
            gdp_df['YoY'] = (gdp_df['VALUE'] - gdp_df['VALUE'].shift(4)) / gdp_df['VALUE'].shift(4) * 100
            gdp_df.dropna(subset=['YoY'], inplace=True)

        total_updated = 0

        # A. FEDFUNDS
        # A. FEDFUNDS
        if 'FEDFUNDS' in data_dict and not data_dict['FEDFUNDS'].empty:
            for _, row in data_dict['FEDFUNDS'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='FEDFUNDS',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        # B. US_CPI_YOY (美國 CPI 年增率)
        if not cpi_df.empty:
            for _, row in cpi_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_CPI_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # C. UNRATE (美國失業率)
        if 'UNRATE' in data_dict and not data_dict['UNRATE'].empty:
            for _, row in data_dict['UNRATE'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='UNRATE',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        # D. NFP_CHANGE (新增非農就業人口，單位：人)
        if not payems_df.empty:
            for _, row in payems_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='NFP_CHANGE',
                    defaults={'value': row['NFP_Change']}
                )
                total_updated += 1

        # E. US_M1_YOY
        if not m1_df.empty:
            for _, row in m1_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_M1_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # F. US_M2_YOY
        if not m2_df.empty:
            for _, row in m2_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_M2_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # G. US_CORE_CPI_YOY
        if not core_cpi_df.empty:
            for _, row in core_cpi_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_CORE_CPI_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # H. US_YIELD_SPREAD (美債利差)
        if 'T10Y2Y' in data_dict and not data_dict['T10Y2Y'].empty:
            for _, row in data_dict['T10Y2Y'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_YIELD_SPREAD',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        # I. US_PCE_YOY
        if not pce_df.empty:
            for _, row in pce_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_PCE_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # J. US_GDP_YOY
        if not gdp_df.empty:
            for _, row in gdp_df.iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_GDP_YOY',
                    defaults={'value': row['YoY']}
                )
                total_updated += 1

        # K. US_SAVING_RATE
        if 'PSAVERT' in data_dict and not data_dict['PSAVERT'].empty:
            for _, row in data_dict['PSAVERT'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_SAVING_RATE',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        # L. US_PERSONAL_INCOME
        if 'PI' in data_dict and not data_dict['PI'].empty:
            for _, row in data_dict['PI'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='US_PERSONAL_INCOME',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        # M. GLOBAL_OIL_PRICE
        if 'MCOILWTICO' in data_dict and not data_dict['MCOILWTICO'].empty:
            for _, row in data_dict['MCOILWTICO'].iterrows():
                _, created = MacroUS.objects.update_or_create(
                    date=row['DATE'].date(),
                    metric='GLOBAL_OIL_PRICE',
                    defaults={'value': row['VALUE']}
                )
                total_updated += 1

        return total_updated

    def _fetch_tw_data(self) -> int:
        """
        從台灣中央銀行 API 下載台灣總經數據並更新資料庫
        """
        # A. 抓取 M1B & M2 年增率 (EF15M01)
        m1b_m2_url = 'https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=EF15M01'
        try:
            r = requests.get(m1b_m2_url, verify=False, timeout=30)
            r.encoding = 'utf-8'
            m_data = r.json()
            datasets = m_data.get('data', {}).get('dataSets', [])
        except Exception as e:
            raise Exception(f"Failed to fetch TW M1B/M2 from CBC API: {e}")

        total_updated = 0
        for item in datasets:
            date_str = item[0]
            m1b_val = item[28]
            m2_val = item[30]

            try:
                date_obj = datetime.strptime(date_str, "%YM%m").date()
            except ValueError:
                continue

            # 限定 15 年時間跨度
            if date_obj < datetime(2011, 1, 1).date():
                continue

            if m1b_val != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='M1B_YOY',
                    defaults={'value': float(m1b_val)}
                )
                total_updated += 1

            if m2_val != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='M2_YOY',
                    defaults={'value': float(m2_val)}
                )
                total_updated += 1

        # B. 抓取 台灣 CPI 年增率, 股價指數, 外匯存底 (EF07M01)
        cpi_url = 'https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=EF07M01'
        try:
            r = requests.get(cpi_url, verify=False, timeout=30)
            r.encoding = 'utf-8'
            c_data = r.json()
            c_datasets = c_data.get('data', {}).get('dataSets', [])
        except Exception as e:
            raise Exception(f"Failed to fetch TW CPI/Index from CBC API: {e}")

        for item in c_datasets:
            date_str = item[0]
            stock_val = item[1]
            forex_val = item[4]
            cpi_val = item[6]

            try:
                date_obj = datetime.strptime(date_str, "%YM%m").date()
            except ValueError:
                continue

            # 限定 15 年時間跨度
            if date_obj < datetime(2011, 1, 1).date():
                continue

            # 寫入 CPI YoY
            if cpi_val != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='CPI_YOY',
                    defaults={'value': float(cpi_val)}
                )
                total_updated += 1

            # 寫入 股價指數
            if stock_val != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='TW_STOCK_INDEX',
                    defaults={'value': float(stock_val)}
                )
                total_updated += 1

            # 寫入 外匯存底 (百萬美元除以 1000 轉為十億美元)
            if forex_val != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='TW_FOREX_RESERVE',
                    defaults={'value': float(forex_val) / 1000.0}
                )
                total_updated += 1

        # C. 抓取 準備貨幣 (EF11M01)
        reserve_url = 'https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=EF11M01'
        try:
            r = requests.get(reserve_url, verify=False, timeout=30)
            r.encoding = 'utf-8'
            res_data = r.json()
            res_datasets = res_data.get('data', {}).get('dataSets', [])
        except Exception as e:
            raise Exception(f"Failed to fetch TW Reserve Money from CBC API: {e}")

        for item in res_datasets:
            date_str = item[0]
            # index 6 是準備貨幣。年增率在 index 1 + 6 * 2 + 1 = 14
            val_yoy = item[14]

            try:
                date_obj = datetime.strptime(date_str, "%YM%m").date()
            except ValueError:
                continue

            # 限定 15 年時間跨度
            if date_obj < datetime(2011, 1, 1).date():
                continue

            if val_yoy != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='TW_RESERVE_MONEY_YOY',
                    defaults={'value': float(val_yoy)}
                )
                total_updated += 1

        # D. 抓取 新台幣匯率 (BP01M01)
        fx_url = 'https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=BP01M01'
        try:
            r = requests.get(fx_url, verify=False, timeout=30)
            r.encoding = 'utf-8'
            fx_data = r.json()
            fx_datasets = fx_data.get('data', {}).get('dataSets', [])
        except Exception as e:
            raise Exception(f"Failed to fetch TW Exchange Rate from CBC API: {e}")

        for item in fx_datasets:
            date_str = item[0]
            # index 0 是新台幣 NTD/USD。數值在 index 1
            val_fx = item[1]

            try:
                date_obj = datetime.strptime(date_str, "%YM%m").date()
            except ValueError:
                continue

            # 限定 15 年時間跨度
            if date_obj < datetime(2011, 1, 1).date():
                continue

            if val_fx != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='TW_EXCHANGE_RATE',
                    defaults={'value': float(val_fx)}
                )
                total_updated += 1

        # E. 抓取 放款與投資量年增率 (EF10M01)
        loan_url = 'https://cpx.cbc.gov.tw/API/DataAPI/Get?FileName=EF10M01'
        try:
            r = requests.get(loan_url, verify=False, timeout=30)
            r.encoding = 'utf-8'
            loan_data = r.json()
            loan_datasets = loan_data.get('data', {}).get('dataSets', [])
        except Exception as e:
            raise Exception(f"Failed to fetch TW Loan/Investment from CBC API: {e}")

        for item in loan_datasets:
            date_str = item[0]
            # index 6 是放款與投資，年增率在 index 1 + 6 * 2 + 1 = 14
            val_yoy = item[14]

            try:
                date_obj = datetime.strptime(date_str, "%YM%m").date()
            except ValueError:
                continue

            # 限定 15 年時間跨度
            if date_obj < datetime(2011, 1, 1).date():
                continue

            if val_yoy != '-':
                _, created = MacroTW.objects.update_or_create(
                    date=date_obj,
                    metric='TW_LOAN_YOY',
                    defaults={'value': float(val_yoy)}
                )
                total_updated += 1

        return total_updated
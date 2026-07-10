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
            'CPILFESL': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPILFESL',  # 核心 CPI 指數
            'T10Y2Y': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y'       # 10Y-2Y 美債利差
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
                raise Exception(f"Failed to fetch US metric {metric} from FRED: {e}")

        # 1. 美國 CPI YoY 年增率計算
        cpi_df = data_dict['CPIAUCSL'].copy()
        cpi_df['YoY'] = (cpi_df['VALUE'] - cpi_df['VALUE'].shift(12)) / cpi_df['VALUE'].shift(12) * 100
        cpi_df.dropna(subset=['YoY'], inplace=True)

        # 2. 非農就業人口按月新增變動 (NFP Change) 計算
        payems_df = data_dict['PAYEMS'].copy()
        payems_df['NFP_Change'] = (payems_df['VALUE'] - payems_df['VALUE'].shift(1)) * 1000  # 轉為人數
        payems_df.dropna(subset=['NFP_Change'], inplace=True)

        # 3. 美國 M1 YoY 年增率計算
        m1_df = data_dict['M1SL'].copy()
        m1_df['YoY'] = (m1_df['VALUE'] - m1_df['VALUE'].shift(12)) / m1_df['VALUE'].shift(12) * 100
        m1_df.dropna(subset=['YoY'], inplace=True)

        # 4. 美國 M2 YoY 年增率計算
        m2_df = data_dict['M2SL'].copy()
        m2_df['YoY'] = (m2_df['VALUE'] - m2_df['VALUE'].shift(12)) / m2_df['VALUE'].shift(12) * 100
        m2_df.dropna(subset=['YoY'], inplace=True)

        # 5. 美國核心 CPI YoY 年增率計算
        core_cpi_df = data_dict['CPILFESL'].copy()
        core_cpi_df['YoY'] = (core_cpi_df['VALUE'] - core_cpi_df['VALUE'].shift(12)) / core_cpi_df['VALUE'].shift(12) * 100
        core_cpi_df.dropna(subset=['YoY'], inplace=True)

        total_updated = 0

        # A. FEDFUNDS
        for _, row in data_dict['FEDFUNDS'].iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='FEDFUNDS',
                defaults={'value': row['VALUE']}
            )
            total_updated += 1

        # B. US_CPI_YOY (美國 CPI 年增率)
        for _, row in cpi_df.iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='US_CPI_YOY',
                defaults={'value': row['YoY']}
            )
            total_updated += 1

        # C. UNRATE (美國失業率)
        for _, row in data_dict['UNRATE'].iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='UNRATE',
                defaults={'value': row['VALUE']}
            )
            total_updated += 1

        # D. NFP_CHANGE (新增非農就業人口，單位：人)
        for _, row in payems_df.iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='NFP_CHANGE',
                defaults={'value': row['NFP_Change']}
            )
            total_updated += 1

        # E. US_M1_YOY
        for _, row in m1_df.iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='US_M1_YOY',
                defaults={'value': row['YoY']}
            )
            total_updated += 1

        # F. US_M2_YOY
        for _, row in m2_df.iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='US_M2_YOY',
                defaults={'value': row['YoY']}
            )
            total_updated += 1

        # G. US_CORE_CPI_YOY
        for _, row in core_cpi_df.iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='US_CORE_CPI_YOY',
                defaults={'value': row['YoY']}
            )
            total_updated += 1

        # H. US_YIELD_SPREAD (美債利差)
        for _, row in data_dict['T10Y2Y'].iterrows():
            _, created = MacroUS.objects.update_or_create(
                date=row['DATE'].date(),
                metric='US_YIELD_SPREAD',
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

        return total_updated
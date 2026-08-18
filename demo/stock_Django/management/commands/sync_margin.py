import sys
from django.core.management.base import BaseCommand
from stock_Django.sync_margin_cli import (
    fetch_twse_margin,
    fetch_tpex_margin,
    save_margin_data_to_db,
    sync_all_margin_data
)

class Command(BaseCommand):
    help = "透過 twse-cli 與 tpex-cli 同步上市與上櫃股票融資融券信用交易資料至 MySQL stock_margin_balance 表"

    def add_arguments(self, parser):
        parser.add_argument(
            '--twse',
            action='store_true',
            help='僅同步 TWSE (上市) 融資融券資料',
        )
        parser.add_argument(
            '--tpex',
            action='store_true',
            help='僅同步 TPEx (上櫃) 融資融券資料',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            default=False,
            help='同步全市場 (上市 + 上櫃) 融資融券資料 (預設)',
        )
        parser.add_argument(
            '--date',
            type=str,
            help='指定同步目標日期 (格式 YYYY-MM-DD)，預設為市場最新交易日',
        )

    def handle(self, *args, **options):
        is_twse = options['twse']
        is_tpex = options['tpex']
        is_all = options['all'] or (not is_twse and not is_tpex)
        target_date = options.get('date')

        self.stdout.write(self.style.NOTICE("=== 開始執行融資融券 CLI 數據同步作業 ==="))
        
        twse_saved = 0
        tpex_saved = 0

        if is_all or is_twse:
            self.stdout.write(">> 正在透過 twse-cli 同步 TWSE (上市) 信用交易數據...")
            twse_data = fetch_twse_margin(target_date=target_date)
            if twse_data:
                twse_saved = save_margin_data_to_db(twse_data)
                self.stdout.write(self.style.SUCCESS(f"[OK] TWSE 上市融資融券寫入成功：{twse_saved} 筆"))
            else:
                self.stdout.write(self.style.WARNING("[!] TWSE 上市無回傳資料或解析失敗"))

        if is_all or is_tpex:
            self.stdout.write(">> 正在透過 tpex-cli 同步 TPEx (上櫃) 信用交易數據...")
            tpex_data = fetch_tpex_margin()
            if tpex_data:
                tpex_saved = save_margin_data_to_db(tpex_data)
                self.stdout.write(self.style.SUCCESS(f"[OK] TPEx 上櫃融資融券寫入成功：{tpex_saved} 筆"))
            else:
                self.stdout.write(self.style.WARNING("[!] TPEx 上櫃無回傳資料或解析失敗"))

        total_saved = twse_saved + tpex_saved
        self.stdout.write(self.style.SUCCESS(f"=== 同步完成！全市場共寫入/更新 {total_saved} 筆 (TWSE: {twse_saved}, TPEx: {tpex_saved}) ==="))

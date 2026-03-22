import os
import logging
import pkgutil
from getpass import getpass


# Get an instance of a logger
logger = logging.getLogger(__name__)

__version__ = '1.1.2'


class LoginPanel():

    def __init__(self):
        pass

    def gui_supported(self):
        try:
            if "VSCODE_PID" in os.environ or pkgutil.find_loader('IPython') is None:
                return False # vscode not support getpass and display at the same time
            else:
                return True
        except:
            return False

    def display_gui(self):

        from IPython.display import IFrame, display, clear_output
        iframe = IFrame(
            f'https://ai.finlab.tw/api_token/?version={__version__}', width=620, height=300)
        display(iframe)

        try:
            token = getpass('請從 https://ai.finlab.tw/api_token 複製驗證碼: \n')
        except:
            print('請從 https://ai.finlab.tw/api_token 複製驗證碼: \n')
            token = input('驗證碼：')
        clear_output()
        self.login(token)

    def display_text_input(self):
        print('請從 https://ai.finlab.tw/api_token 複製驗證碼，貼於此處:\n')
        token = input('驗證碼：')
        self.login(token)
        print('之後可以使用以下方法自動登入')
        print('import finlab')
        print('finlab.login("YOUR API TOKEN")')

    @staticmethod
    def login(token):
        # set token
        token = token[:64]
        os.environ['finlab_id_token'] = token
        os.environ['FINLAB_API_TOKEN'] = token
        print('輸入成功!')


def login(api_token=None):
    """登錄量化平台。

    可以至 [api_token查詢頁面](https://ai.finlab.tw/api_token/) 獲取api_token，傳入函數後執行登錄動作。
    之後使用Finlab模組的會員功能時，系統就不會自動跳出請求輸入api_token的[GUI頁面](https://ai.finlab.tw/api_token/)。
    若傳入的api_toke格式有誤，系統會要求再次輸入。

    Args:
        api_token (str): FinLab api_token
    """
    lp = LoginPanel()

    if api_token:
        lp.login(api_token)
        return

    if lp.gui_supported():
        lp.display_gui()
    else:
        lp.display_text_input()


def get_token():
    """取得登錄會員的finlab_id。

    若未登錄過，會跳出登錄頁面請求登錄。

    Returns:
        (str): finlab api token
    """
    if 'FINLAB_API_TOKEN' not in os.environ:
        login()

    return os.environ['FINLAB_API_TOKEN'][:64]

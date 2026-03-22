import os
import re
import sys
import json
import logging
import tempfile
from pathlib import Path
import finlab
from finlab.core.utils_core import str_to_bytearray

# Get an instance of a logger
logger = logging.getLogger(__name__)

def requests_module_factory():

    """Creates a requests module

    The function returns requests module if it is installed.
    If requests module is not installed, raise error to inform user.
    If the python version is pyodide, try to create a custom requests module.

    Returns:
        requests module

    """

    # check requests is installed and return the module
    if "pyodide" not in sys.modules:
        import requests
        return requests

    # raise error if not install requests and not in pyodide env
    from js import XMLHttpRequest, Blob, FormData
    import urllib.parse
    from pyodide import http
    import pyodide

    # in pyodide env, make fake requests
    class Response:
        text = ''
        json_ = ''
        content = ''
        ok = True

        def json(self):
            return self.json_

    class requests:

        def getBytes(url):
            req = XMLHttpRequest.new()
            req.open("GET", url, False)
            req.overrideMimeType('text/plain; charset=x-user-defined')
            req.send(None)
            return str_to_bytearray(req.response)

        def get(url, params=None, **kwargs):

            if params == None:
                params = {}


            complete_url = url + '?' + '&'.join([
                urllib.parse.quote_plus(k)
                + '=' + urllib.parse.quote_plus(v)
                for k, v in params.items()])

            str_io = http.open_url(complete_url)

            res = Response()
            res.text = str_io.read()
            res.json_ = requests.parse_json(res.text)
            res.status_code = 200

            return res

        def post(url, data, asyn=False, **kwargs):

            if data == None:
                data = {}


            req = XMLHttpRequest.new()
            form = FormData.new()

            for k, v in data.items():
                form.append(k, v)

            req.open("POST", url, asyn)
            req.send(form)

            res = Response()
            res.content = req.response
            res.text = str(req.response)
            res.json_ = requests.parse_json(res.text)
            res.status_code = 200

            return res

        def parse_json(text):

            if len(text) == 0:
                return {}

            if text[-1] == '\n':
                text = text[:-1]

            if text[0] != '{' or text[-1] != '}':
                return {}

            try:
                return json.loads(text)
            except:
                return {}

    return requests

import os
import tempfile
from pathlib import Path

def get_tmp_dir():
    # Check if running in Google Colab
    if 'COLAB_GPU' in os.environ:
        # Try to use Google Drive as the tmp directory
        try:
            from google.colab import drive

            # Mount Google Drive
            if not os.path.exists('/content/drive/'):
                drive.mount('/content/drive')

            # Custom tmp dir inside Google Drive
            google_drive_tmp_dir = Path("/content/drive/My Drive/Colab_tmp/finlab_db")
            google_drive_tmp_dir.mkdir(parents=True, exist_ok=True)

            return str(google_drive_tmp_dir)

        except ImportError:
            print("Google Drive is not mounted. Please mount Google Drive to use as tmp directory.")
            raise

    # If running in a cloud function environment
    elif os.environ.get('FUNCTION_TARGET', None):
        # Set tmp directory to be /tmp for cloud functions
        tmp_subdir = Path("/tmp/finlab_db")
        tmp_subdir.mkdir(parents=True, exist_ok=True)
        return str(tmp_subdir)


    # For local machine use-cases
    else:
        # Default tmp directory
        home_dir = Path.home()

        # Create 'finlab_db' under tmp directory
        tmp_subdir = Path(home_dir) / 'finlab_db'
        tmp_subdir.mkdir(parents=True, exist_ok=True)

        return str(tmp_subdir)



def check_version_function_factory():

    """ Check finlab package version is the latest or not

    if the package version is out of date, info user to update.

    Returns
        None

    """
    if "pyodide" in sys.modules:
        return lambda : None

    latest_package_version = None

    def ret():

        nonlocal latest_package_version

        if latest_package_version is None:
            res = requests.get('https://pypi.org/project/finlab/')
            res.encoding = 'utf-8'

            m = re.findall("finlab\s([a-z0-9.]*)\s*</h1>", res.text)
            latest_package_version =  m[0] if m else None

            if latest_package_version != finlab.__version__:
                logger.warning(f'Your version is {finlab.__version__}, please install a newer version.\nUse "pip install finlab=={latest_package_version}" to update the latest version.')

    return ret


def global_object_getter_setter_factory():
    _finlab_global_objects = {}

    def get_global(name):
        nonlocal _finlab_global_objects
        if name in _finlab_global_objects:
            return _finlab_global_objects[name]
        else:
            return None

    def set_global(name, obj):
        nonlocal _finlab_global_objects
        _finlab_global_objects[name] = obj
        if "pyodide" in sys.modules and "js" in sys.modules:
            import js
            from pyodide import ffi
            if hasattr(js, "_pyodide_execution_id"):
                js.postMessage(ffi.to_js({'content': obj, 'finish': False, 'type': name, 'id': js._pyodide_execution_id}))


    return get_global, set_global

def notify_progress(snippet_id, position, progress):
    """  notify progress of the backtest (only work in pyodide)
    """

    if "pyodide" in sys.modules and "js" in sys.modules:
        import js
        from pyodide import ffi
        if hasattr(js, "_pyodide_execution_id"):
            content = {
                "nstocks": (position!=0).sum(axis=1).mean(),
                "timestamp": snippet_id,
                "progress": progress,
            }

            js.postMessage(ffi.to_js({'content': content, 'finish': False, 'type': "backtest-progress", 'id': js._pyodide_execution_id}))


get_global, set_global = global_object_getter_setter_factory()

requests = requests_module_factory()
check_version = check_version_function_factory()


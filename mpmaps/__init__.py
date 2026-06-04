__author__ = """Nicolas Aunai"""
__email__ = "nicolas.aunai@lpp.polytechnique.fr"
__version__ = '0.2.0'

import os
import sys

from .mpmaps import MPMap
from .globals import grids


_IN_PYODIDE = "pyodide" in sys.modules
_SKIP_DOWNLOAD = bool(os.environ.get("MPMAPS_SKIP_DOWNLOAD"))

if not _IN_PYODIDE and not _SKIP_DOWNLOAD:
    import urllib.request
    from platformdirs import user_data_dir

    data_dir = os.path.join(user_data_dir(), "mpmaps")
    base_url = "https://hephaistos.lpp.polytechnique.fr/data/mpmaps_grids"
    grid_urls = {g: base_url + "/" + g for g in grids}

    for grid, url in grid_urls.items():
        dlpath = os.path.join(data_dir, grid)
        if not os.path.isfile(dlpath):
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            print(f"file {dlpath} does not exist, downloading {url} --> {dlpath}\n")
            urllib.request.urlretrieve(url, dlpath)

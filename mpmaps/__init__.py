__author__ = """Nicolas Aunai"""
__email__ = "nicolas.aunai@lpp.polytechnique.fr"
__version__ = '0.2.0'

from platformdirs import user_data_dir
import urllib.request
import os

from .mpmaps import MPMap
from .globals import grids


data_dir = os.path.join(user_data_dir(), "mpmaps")


base_url = "https://hephaistos.lpp.polytechnique.fr/data/mpmaps_grids"

grid_urls = {g: base_url + "/" + g for g in grids}


for grid, url in grid_urls.items():

    dlpath = os.path.join(data_dir, grid)
    exist = os.path.isfile(dlpath)

    if not exist:

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(f"file {dlpath} does not exist, downloading {url} --> {dlpath}\n")
        urllib.request.urlretrieve(url, dlpath)

import logging
import tempfile
from pathlib import Path
from zipfile import ZipFile

import click
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from rich import print as rprint

import challenge.data.io as io
import challenge.data.paths as paths

log = logging.getLogger("challenge")


def clean():
    """Removes all data files."""
    for file in paths.data_files:
        if file.exists():
            file.unlink()
            log.info(
                f"[red]❌ removed[/] file [cyan]{io.repr(file)}[/]",
                extra={"markup": True},
            )


def download():
    """Downloads data files from Kaggle."""
    download_base = "mva-mash-kernel-methods-2021-2022"

    api = KaggleApi()
    api.authenticate()
    log.info("[green]🔑 authentification successful[/]", extra={"markup": True})

    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = Path(_temp_dir)
        try:
            api.competition_download_files(download_base, path=temp_dir)
            log.info(
                f"[green]🔗 downloaded[/] zip archive from Kaggle [cyan]{download_base}[/]",
                extra={"markup": True},
            )
        except Exception:
            log.error(
                f"[red]Could not download from Kaggle [cyan]{download_base}[/]",
                extra={"markup": True},
            )
            raise

        with ZipFile(temp_dir / f"{download_base}.zip") as zip_file:
            zip_file.extractall(paths.data_dir)
            log.info(
                f"[green]📦 extracted[/] archive to [cyan]{io.repr(paths.data_dir)}[/]",
                extra={"markup": True},
            )


def prepare():
    """Data files pre-processing."""
    for file in paths.x_downloaded:
        if not file.exists():
            log.error(
                f"[red]Missing files[/] [cyan]{io.repr(file)}[/] try downloading them again.",
                extra={"markup": True},
            )
            raise FileNotFoundError(file)

        # array has one spurious dimension because of trailing commas
        array = np.genfromtxt(file, delimiter=",", usecols=range(3072))
        io.dumpx(array, out=file)
        log.info(
            f"[green]🧹 cleaned[/] data file [cyan]{io.repr(file)}[/]",
            extra={"markup": True},
        )


def split(proportion):
    """Splits train set in train/val."""
    data = io.xload(paths.x_train)
    len_train = (len(data) * proportion) // 100
    io.dumpx(data[:len_train], out=paths.x_train)
    io.dumpx(data[len_train:], out=paths.x_val)
    log.info("[green]✂️ Split[/] X into train/val datasets", extra={"markup": True})

    data = io.yload(paths.y_train)
    io.dumpy(data[:len_train], out=paths.y_train)
    io.dumpy(data[len_train:], out=paths.y_val)
    log.info("[green]✂️ Split[/] Y into train/val datasets", extra={"markup": True})

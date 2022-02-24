import tempfile
from pathlib import Path
from zipfile import ZipFile

import challenge.data.io as io
import challenge.data.paths as paths
import click
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from rich import print as rprint


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Regenerates data files."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(clean)
        ctx.invoke(download)
        ctx.invoke(prepare)
        ctx.invoke(val)


@cli.command()
def clean():
    """Removes all data files."""
    for file in paths.data_files:
        if file.exists():
            file.unlink()
            rprint(f"[red]❌ removed[/] file [cyan]{file.name}[/]")


@cli.command()
def download():
    """Downloads data files from Kaggle."""
    download_base = "mva-mash-kernel-methods-2021-2022"

    api = KaggleApi()
    api.authenticate()
    rprint("[green]🔑 authentification[/] successful")

    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = Path(_temp_dir)
        try:
            api.competition_download_files(download_base, path=temp_dir)
            rprint(
                f"[green]🔗 downloaded[/] zip archive from Kaggle [cyan]{download_base}[/]"
            )
        except Exception:
            rprint(f"[red]Could not download from Kaggle [cyan]{download_base}[/]")
            raise

        with ZipFile(temp_dir / f"{download_base}.zip") as zip_file:
            zip_file.extractall(paths.data_dir)
            rprint(f"[green]📦 extracted[/] archive to [cyan]{paths.data_dir}[/]")


@cli.command()
def prepare():
    """Data files pre-processing."""
    for file in paths.x_downloaded:
        if not file.exists():
            rprint("[red]Missing files[/] try downloading them again.")
            raise FileNotFoundError(file)

        # array has one spurious dimension because of trailing commas
        array = np.genfromtxt(file, delimiter=",", usecols=range(3072))
        io.xdump(array, out=file)
        rprint(f"[green]🧹 cleaned[/] data file [cyan]{file.name}[/]")


@cli.command()
def val():
    """Splits train set in train/val."""
    data = io.xload(paths.x_train)
    len_train = (len(data) * 80) // 100
    io.xdump(data[:len_train], out=paths.x_train)
    io.xdump(data[len_train:], out=paths.x_val)
    rprint("[green]✂️ Split[/] X into train/val datasets")

    data = io.yload(paths.y_train)
    io.ydump(data[:len_train], out=paths.y_train)
    io.ydump(data[len_train:], out=paths.y_val)
    rprint("[green]✂️ Split[/] Y into train/val datasets")


if __name__ == "__main__":
    cli()

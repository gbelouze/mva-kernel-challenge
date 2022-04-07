import logging
from pathlib import Path

import click
from rich.logging import RichHandler

import challenge.data.make as data_make

log = logging.getLogger("challenge")


def setup_log():
    FORMAT = "%(message)s"
    logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
    log.setLevel(logging.INFO)


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("--quiet/--no-quiet", default=False)
def main(verbose, quiet):
    setup_log()
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    if quiet:
        log.setLevel(logging.ERROR)


@main.group()
def data():
    pass


@data.command()
@click.option(
    "-p",
    "--proportion",
    type=int,
    help="Proportion of train data in percent",
    default=80,
)
def make(proportion):
    """Downloads and prepares data files from Kaggle."""
    assert 0 <= proportion <= 100, "Not a valid percentage"
    data_make.download()
    data_make.prepare()
    data_make.split(proportion)


@data.command()
def clean():
    """Removes all data files."""
    data_make.clean()

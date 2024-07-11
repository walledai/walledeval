# walledeval/cli/data/datasets.py

import typer

datasets_app = typer.Typer()


@datasets_app.command()
def bye():
    print("bye")

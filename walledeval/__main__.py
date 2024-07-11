# walledeval/__main__.py

import typer

from cli.data.datasets import datasets_app

app = typer.Typer()
app.add_typer(datasets_app, name="datasets")

if __name__ == "__main__":
    app()

# Installation

## Installing from PyPI

Yes, we have published WalledEval on PyPI! To install WalledEval and all its dependencies, the easiest method would be to use 
`pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following 
command in a terminal or Command Prompt / Powershell:

```bash
$ pip install walledeval
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the
following command too:

```bash
$ python -m pip install walledeval
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation 
configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

## Installing from Source

To install from source, you need to get the following:

### Git

Git is needed to install this repository. This is not completely necessary as you can also install the zip file for this 
repository and store it on a local drive manually. To install Git, follow 
[this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt etc:

```bash
$ git clone https://github.com/walledai/walledeval.git
```

This stores a copy in the folder `walledeval`. You can then navigate into it using `cd walledeval`.

### Poetry

This project can be used easily via a tool known as Poetry. This allows you to easily reflect edits made in the original 
source code! To install `poetry`, you can also install it using `pip` by typing in the command as follows:

```bash
$ pip install poetry
```

Again, if you have any issues with `pip`, check out [here](#installing-from-pypi).

After this, you can use the following command to install this library:

```bash
$ poetry install
```

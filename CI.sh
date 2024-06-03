#!/usr/bin/env sh

# pip3 install -r requirements.txt

git pull
mkdocs build
mkdocs gh-deploy
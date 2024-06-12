#!/bin/bash

# get date
slug=$(date '+%y%m%d%H%M')

# create article
npx zenn new:article --slug "$slug"_hoge --title タイトル --type tech --emoji 🍀

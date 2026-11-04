---
title: robolint
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Quality checks for LeRobot robot learning datasets
tags:
  - robotics
  - lerobot
  - data-quality
---

# robolint

Browse the results of an audit of 18 public LeRobot datasets, or try an experimental check of any public LeRobot dataset that runs entirely in your browser.

Source, command-line tool and full write-up: https://github.com/maxmccutcheon59/robolint

## Updating this Space

The files here are generated from the robolint repo:

```bash
python space_static/build.py path/to/audit   # writes results.json and copies the robolint package
hf upload maxmccutcheon59/robolint space_static . --repo-type space
```

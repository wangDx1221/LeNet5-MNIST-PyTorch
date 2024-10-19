# -*- coding: utf-8 -*-
# @Time     : 2024/10/16 21:11
# @Author   : wangDx
# @File     : main.py
# @describe : py脚本依次运行多个文件

import subprocess

subprocess.run(["python", "model.py"])

subprocess.run(["python", "train.py"])

subprocess.run(["python", "test.py"])

subprocess.run(["python", "example.py"])

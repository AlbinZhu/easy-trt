import sys
import os

print(os.getenv("PATH"))

print(sys.path)
print(sys.executable)

import cv2

print(cv2.__version__)

for i in range(1, len(sys.argv)):
    print("参数 %s 为：%s" % (i, sys.argv[i]))

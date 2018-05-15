import time
import json
import sys
import io

i = 0
counter = 0

while i < 100000000:
    time.sleep(1)
    counter = counter + 1
    i = i + 1

    # flush=True, siehe: https://stackoverflow.com/questions/25607799/node-jss-python-child-script-outputting-on-finish-not-real-time
    # print(json_data, flush=True);

    print(counter);

lines = sys.stdin.readlines()
print(lines)
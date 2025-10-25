#!/usr/bin/env python

import sys, os, subprocess
from pathlib import Path

if len(sys.argv) != 11:
    print("The sertain arguments should be specified:")
    print("    1. 'PATH_TO_LIVEPROCMON_TEMPLATE_FILE'")
    print("    2. 'PATH_TO_EXECUTABLE'")
    print("    3. 'PATH_TO_BUILD_FOLDER'")
    print("    4. 'PORT'")
    print("    5. 'MAX_POINTS_TO_COLLECT'")
    print("    6. 'COLLECT_DELAY_MS'")
    print("    7. 'GENERATE_PLOT_DELAY_MS'")
    print("    8. 'UPDATE_FREQUECY_MS'")
    print("    9. 'PLOT_WIDTH_INCH'")
    print("   10. 'PLOT_HEIGHT_INCH'")
    sys.exit(1)

PATH_TO_LIVEPROCMON_TEMPLATE_FILE = sys.argv[1]
PATH_TO_EXECUTABLE = sys.argv[2] 
PATH_TO_BUILD_FOLDER = sys.argv[3] 
PORT = int(sys.argv[4])
MAX_POINTS_TO_COLLECT = int(sys.argv[5])
COLLECT_DELAY_MS = int(sys.argv[6])
GENERATE_PLOT_DELAY_MS = int(sys.argv[7])
UPDATE_FREQUECY_MS = int(sys.argv[8])
PLOT_WIDTH_INCH = float(sys.argv[9])
PLOT_HEIGHT_INCH = float(sys.argv[10])

with open(PATH_TO_LIVEPROCMON_TEMPLATE_FILE, 'r') as file:
    liveprocmon_cpumem_filedata = file.read()
liveprocmon_cpumem_filedata = liveprocmon_cpumem_filedata.replace('LIVEPROCMON_TEMPLATE_PATH_TO_EXECUTABLE', PATH_TO_EXECUTABLE)
liveprocmon_cpumem_filedata = liveprocmon_cpumem_filedata.replace('LIVEPROCMON_TEMPLATE_MAX_POINTS_TO_COLLECT', str(MAX_POINTS_TO_COLLECT))
liveprocmon_cpumem_filedata = liveprocmon_cpumem_filedata.replace('LIVEPROCMON_TEMPLATE_UPDATE_FREQUECY_MS', str(UPDATE_FREQUECY_MS))
liveprocmon_cpumem_filedata = liveprocmon_cpumem_filedata.replace('LIVEPROCMON_TEMPLATE_PLOT_WIDTH_INCH', str(PLOT_WIDTH_INCH))
liveprocmon_cpumem_filedata = liveprocmon_cpumem_filedata.replace('LIVEPROCMON_TEMPLATE_PLOT_HEIGHT_INCH', str(PLOT_HEIGHT_INCH))
Path('generated').mkdir(parents=True, exist_ok=True)
with open('generated/liveprocmon.py', 'w') as file:
    file.write(liveprocmon_cpumem_filedata)


os.chdir('generated')
import time, generated.liveprocmon as liveprocmon

liveprocmon.generate_index_html()


from threading import Thread

COLLECT_DELAY_S = float(COLLECT_DELAY_MS)/float(1000)
def collect_data():
    while True:
        liveprocmon.collect_data()
        time.sleep(COLLECT_DELAY_S)
thread_collect_data = Thread(target = collect_data)
thread_collect_data.start()

GENERATE_PLOT_DELAY_S = float(GENERATE_PLOT_DELAY_MS)/float(1000)
while True:
    time.sleep(GENERATE_PLOT_DELAY_S)
    liveprocmon.generate_plot_image()

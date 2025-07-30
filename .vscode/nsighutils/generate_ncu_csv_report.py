import os
import sys
import subprocess

command = "ncu"
report_path= "{0}.csv".format(sys.argv[2])

with open(report_path, 'w') as report_file:
    report_file.write(subprocess.check_output("ncu --metrics '{0}' --csv '{1}'".format(sys.argv[1], sys.argv[2]), shell=True, text=True))

os.system("code -r '{0}'".format(report_path))

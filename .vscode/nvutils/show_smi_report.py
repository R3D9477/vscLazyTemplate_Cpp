import os
import subprocess

report_path = ".nvidia-smi"

with open(report_path, 'w') as report_file:
    report_file.write(subprocess.check_output("nvidia-smi", shell=True, text=True))

os.system("code -r '{0}'".format(report_path))

import os
import sys
import subprocess

command = "ncu"
report_path= "{0}.csv".format(sys.argv[2])

report_lines = subprocess.check_output(["ncu","--metrics", sys.argv[1], "--csv", sys.argv[2]]).decode().splitlines()

with open(report_path, 'w') as report_file:
    write_to_file = False
    for line in report_lines:
        if write_to_file:
            if "==WARNING==" not in line:
                report_file.write("%s\n" % line)
        elif "==PROF== Disconnected" in line:
            write_to_file = True

os.system("code -r '{0}'".format(report_path))

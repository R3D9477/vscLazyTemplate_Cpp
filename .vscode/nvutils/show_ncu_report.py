import os
import sys
import subprocess

profiler = sys.argv[1]

report_path= "{0}.csv".format(sys.argv[3])

report_lines = subprocess.check_output([profiler,"--metrics", sys.argv[2], "--csv", sys.argv[3]]).decode().splitlines()

with open(report_path, 'w') as report_file:
    write_to_file = False
    for line in report_lines:
        if write_to_file:
            if "==WARNING==" not in line:
                report_file.write("%s\n" % line)
        elif "==PROF== Disconnected" in line:
            write_to_file = True

os.system("code -r '{0}'".format(report_path))

import os, sys, subprocess

COMPILER_PATH = sys.argv[1]
CPP_STANDRD_N = sys.argv[2]
OPTIMIZATION_LEVEL = sys.argv[3]
EXTRA_ARGS = sys.argv[4]
BUILD_DIRECTORY = sys.argv[5]
SOURCE_FILE = sys.argv[6]

OUTPUT_APP_PATH = os.path.join(BUILD_DIRECTORY, os.path.splitext(os.path.basename(SOURCE_FILE))[0])

HEADER_FOLDERS = [os.path.dirname(SOURCE_FILE)]
SOURCE_FILES = [SOURCE_FILE]

SRC_DIRECTORY = os.path.dirname(SOURCE_FILE)
for cpp_file in os.listdir(SRC_DIRECTORY):
    if cpp_file.endswith(".cc") or cpp_file.endswith(".cpp"):
        cpp_file_path = os.path.join(SRC_DIRECTORY, cpp_file)
        if cpp_file_path != SOURCE_FILE:
            with open(cpp_file_path) as f:
                if 'int main' not in f.read():
                    SOURCE_FILES.append(cpp_file_path)

cc_cmd = '{0} -std=c++{1} -{2} {3} {4} {5} -o {6}'.format(
    COMPILER_PATH,
    CPP_STANDRD_N,
    OPTIMIZATION_LEVEL,
    EXTRA_ARGS,
    " ".join(["-I"+h for h in HEADER_FOLDERS]),
    " ".join(SOURCE_FILES),
    OUTPUT_APP_PATH
)

try:
    cc_output = subprocess.check_output(cc_cmd, stderr=subprocess.STDOUT, shell=True, timeout=3, universal_newlines=True)
except subprocess.CalledProcessError as cc_exc:
    print(cc_exc.output, file=sys.stderr)
    exit(cc_exc.returncode)
else:
    print(cc_output, file=sys.stdout)

exit(0)

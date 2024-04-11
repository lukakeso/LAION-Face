import os
from pathlib import Path
import argparse


if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--dir", default="glasses", help="directory with the .tar.gz files")

    args=parser.parse_args()
    os.chdir(args.dir)
    os.makedirs("data", exist_ok=True)
    for d in os.listdir():
        if d.endswith(".tar") and not(Path(d.split(".")[0]).is_dir()):
            print("Deflating {}".format(d))
            os.system("tar -xf {} -C data".format(d))
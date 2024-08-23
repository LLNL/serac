#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: run_benchmarks.py

 description: 
  Run benchmarks and update shared (or any desired) location with new Caliper files

"""

from common_build_functions import *

from argparse import ArgumentParser

import os


def parse_args():
    "Parses args from command line"
    parser = ArgumentParser()
    parser.add_argument("-sd", "--spot-directory",
                      dest="spot_dir",
                      default=get_shared_spot_dir(),
                      help="Where to put all resulting caliper files to use for SPOT analysis (defaults to a shared location)")
    parser.add_argument("-hc", "--host-config",
                    dest="host_config",
                    default="",
                    required=True,
                    help="Specific host-config file to build")
    parser.add_argument("-t", "--timestamp",
                    dest="timestamp",
                    default=get_timestamp(),
                    help="Set timestamp manually for debugging")
    ###############
    # parse args
    ###############
    args, extra_args = parser.parse_known_args()
    args = vars(args)
    return args


def main():
    args = parse_args()

    # Args
    spot_dir = args["spot_dir"]
    host_config = args["host_config"]
    timestamp = args["timestamp"]

    # Vars
    repo_dir = get_repo_dir()
    test_root = get_build_and_test_root(repo_dir, timestamp)

    # Build Serac
    os.chdir(repo_dir)
    os.makedirs(test_root)
    build_and_test_host_config(test_root, host_config, True, "-DENABLE_BENCHMARKS=ON", True, True, "")

    # Go to build location
    build_dir=""
    dirs = glob.glob(pjoin(test_root, "*"))
    for dir in dirs:
        if os.path.exists(dir) and "build-" in dir:
            build_dir=dir
    os.chdir(build_dir)

    # Run benchmarks
    # TODO Instead of running each benchmark individually, create a custom target
    # shell_exec("make benchmarks -j")
    shell_exec("tests/profiling", print_output=True)
    shell_exec("benchmarks/benchmark_thermal", print_output=True)
    shell_exec("benchmarks/benchmark_functional", print_output=True)

    # Move resulting .cali files to specified directory
    os.makedirs(spot_dir, exist_ok=True)
    cali_files = glob.glob(pjoin(build_dir, "*.cali"))
    print(cali_files)
    for cali_file in cali_files:
        if os.path.exists(cali_file):
            shutil.move(cali_file, spot_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())

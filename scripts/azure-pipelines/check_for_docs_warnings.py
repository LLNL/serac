#!/bin/sh
"exec" "python" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Checks log for warnings and errors")

    parser.add_argument("-l","--log", type=str, required=True, help="Path to log file to be checked") 

    args = parser.parse_args()

    print("~~~~~~~ Given Command line Arguments ~~~~~~~")
    print("Log Path: {0}".format(args.log))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return args


def main():
    args = parse_args()

    with open(args.log, "r") as f:
        log_lines = f.readlines()

    # Get warnings/errors out of log file
    warnings = []
    errors = []
    for log_line in log_lines:
        if "warning:" in log_line.lower():
            warnings.append(log_line)
        elif "error:" in log_line.lower():
            warnings.append(log_line)

    # Print warnings/errors that are found
    if len(warnings) > 0:
        print("~~~~~~~~~~~~~~~ Warnings ~~~~~~~~~~~~~~~")
        for warning in warnings:
            print(warning)
        print("")

    if len(errors) > 0:
        print("~~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~~~~")
        for error in errors:
            print(error)
        print("")

    # Print summary info
    print("~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~")
    print("Warning Count: {0}".format(len(warnings)))
    print("Error Count:   {0}".format(len(errors)))
    total_count = len(warnings) + len(errors)
    print("Total Count:   {0}".format(total_count))

    # Error out if any found
    if total_count > 0:
        return False
    return True

if __name__ == "__main__":
    sys.exit(main())

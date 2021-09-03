#!/bin/sh
"exec" "python" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

import os
import sys

def print_usage():
    print("Usage: " + sys.argv[0] + " <path to documentation build output>")

def handle_command_line():
    # Check for help
    if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
        print_usage()
        sys.exit(0)

    # Check arg count
    if len(sys.argv) < 2:
        print("Error: Too few command line arguments given")
        print_usage()
        sys.exit(1)

    if len(sys.argv) > 2:
        print("Error: Too many command line arguments given")
        print_usage()
        sys.exit(1)

    # Check given file path
    log_path = sys.argv[1]
    if not os.path.isfile(log_path):
        print("Error: Given path is not a file: {0}".format(log_path))
        print_usage()
        sys.exit(1)

    print("Log Path: {0}".format(log_path))

    return log_path


def main():
    log_path = handle_command_line()

    with open(log_path, "r") as f:
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

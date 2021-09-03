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
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Checks log for warnings and errors")

    parser.add_argument("-l", "--log", type=str, required=True, help="Path to log file to be checked")
    parser.add_argument("-i", "--ignore", type=str, required=False, help="Path to file that includes regex's to ignore lines")

    args = parser.parse_args()

    print("~~~~~~~ Given Command line Arguments ~~~~~~~")
    print("Ignore file Path: {0}".format(args.ignore))
    print("Log Path:         {0}".format(args.log))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return args


def main():
    args = parse_args()

    with open(args.log, "r") as f:
        log_lines = f.readlines()

    if not args.ignore:
        ignore_regexs = []
    else:
        with open(args.ignore, "r") as f:
            ignore_regexs = f.readlines()

    # Get warnings/errors out of log file
    ignores = []
    warnings = []
    errors = []
    for log_line in log_lines:
        # First, check if it matches any ignore regex's
        matches_ignore = False
        for ignore_regex in ignore_regexs:
            if re.match(ignore_regex, log_line):
                matches_ignore = True
                break

        lowered = log_line.lower()
        has_warning = "warning:" in log_line.lower()
        has_error = "error:" in log_line.lower()

        if (has_warning or has_error) and matches_ignore:
            ignores.append(log_line)
        elif has_error:
            errors.append(log_line)
        elif has_warning:
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

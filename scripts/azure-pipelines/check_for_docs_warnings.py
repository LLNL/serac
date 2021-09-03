#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"
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
    print("")

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
        # First, check if it is a warning/error
        lowered = log_line.lower()
        has_warning = "warning:" in lowered
        has_error = "error:" in lowered

        # Then, check if it matches any ignore regex's
        matches_ignore = False
        if has_error or has_warning:
            for ignore_regex in ignore_regexs:
                if re.search(ignore_regex, log_line):
                    matches_ignore = True
                    break

        if matches_ignore:
            ignores.append(log_line)
        elif has_error:
            errors.append(log_line)
        elif has_warning:
            warnings.append(log_line)

    # Print ignores/warnings/errors that are found
    print("~~~~~~~~~~~~~~~ Ignores ~~~~~~~~~~~~~~~~")
    for ignore in ignores:
        print(ignore)
    print("")

    print("~~~~~~~~~~~~~~~ Warnings ~~~~~~~~~~~~~~~")
    for warning in warnings:
        print(warning)
    print("")

    print("~~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~~~~")
    for error in errors:
        print(error)
    print("")

    # Print summary info
    print("~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~")
    print("Warning Count: {0}".format(len(warnings)))
    print("Error Count:   {0}".format(len(errors)))
    print("Ignored Count: {0}".format(len(ignores)))

    # Error out if any found
    # TODO: uncomment this in follow up PR that fixes warnings/errors
    # if (len(warnings) + len(errors)) > 0:
    #     return False
    return True

if __name__ == "__main__":
    sys.exit(main())

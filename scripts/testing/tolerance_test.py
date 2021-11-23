#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

import argparse
import json
import math
import os
import sys


# This script compares two Serac summary files against each other.
# The 'baseline' file is treated as a truth against the possibly wrong
# 'test' file. It attempts to fail with the most error messages it can at a given
# correctness level (field names > stat names > stat values).
# Summary files were the same within the given tolerance if the
# script exists successfully and not otherwise.

def ensure_file(path):
    if not os.path.exists(path):
        print("ERROR: Given file does not exist: {0}".format(path))
        sys.exit(1)
    if not os.path.isfile(path):
        print("ERROR: Given file is not a file: {0}".format(path))
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two Serac summary files")

    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline summary file")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to test summary file")
    parser.add_argument("--tolerance", type=float, required=True,
                        help="Allowed tolerance amount for individual values")

    args = parser.parse_args()

    # Ensure correctness of given options
    ensure_file(args.baseline)
    ensure_file(args.test)

    # Print options
    print("------- Given Options -------")
    print("Baseline file: {0}".format(args.baseline))
    print("Test file:     {0}".format(args.test))
    print("Tolerance:     {0}".format(args.tolerance))
    print("-----------------------------")

    return args


# Ensure that time steps exist and the same in both files
def ensure_timesteps(baseline_curves, test_curves):
    timestep_name = 't'
    error_found = False
    # Check if both files have t
    if not timestep_name in baseline_curves.keys():
        print("ERROR: Could not find timesteps, {0}, in baseline file".format(timestep_name))
        error_found = True
    if not timestep_name in test_curves.keys():
        print("ERROR: Could not find timesteps, '{0}', in test file".format(timestep_name))
        error_found = True

    if error_found:
        sys.exit(1)

    baseline_timesteps = baseline_curves[timestep_name]
    test_timesteps = test_curves[timestep_name]

    if len(baseline_timesteps) != len(test_timesteps):
        print("ERROR: Number of test time steps, {0}, does not match baseline, {1}"
              .format(len(test_timesteps), len(baseline_timesteps)))
        sys.exit(1)

    for i in range(len(baseline_timesteps)):
        if baseline_timesteps[i] != test_timesteps[i]:
            print("ERROR: Test file time step {0} did not match baseline: {1} vs {2}"
                  .format(i, baseline_timesteps[i], test_timesteps[i]))
            error_found = True

    if error_found:
        sys.exit(1)


 # filter field names from a list of all possible names
def get_field_names(possible_fields):
    field_names = []
    # known non-field names that should be ignored
    non_field_names = ["t"]
    for key in possible_fields.keys():
        if key in non_field_names:
            continue
        field_names.append(key)
    return field_names


# Return list of items in l1 that are missing from l2
def list_missing(l1, l2):
    missing = [x for x in l1 if x not in l2]
    return missing


# Output the missing/extra fields in the test list and return True on same
def output_missing(baseline_list, test_list, type, field_name=""):
    baseline_missing = list_missing(test_list, baseline_list)
    test_missing = list_missing(baseline_list, test_list)

    if (len(baseline_missing) > 0) or (len(test_missing) > 0):
        if not field_name:
            print("ERROR: Test and baseline files have mismatching {0}.".format(type))
        else:
            print("ERROR: Test and baseline files have mismatching {0} in field '{1}'.".format(type, field_name))

        if len(test_missing) > 0:
            print("       Missing {0} in test file: {1}".format(type, ", ".join(test_missing)))
        if len(baseline_missing) > 0:
            print("       Extra {0} in test file: {1}".format(type, ", ".join(baseline_missing)))
        return False

    return True


# Ensure list of field names are equal and error out with useful message
def ensure_field_names(baseline_field_names, test_field_names):
    error_found = False

    # Check if either file have no fields
    if len(baseline_field_names) == 0:
        print("ERROR: Baseline file had no field names")
        error_found = True
    if len(test_field_names) == 0:
        print("ERROR: Test file had no field names")
        error_found = True

    if error_found:
        sys.exit(1)

    # If fields are present, check for extra/missing fields in test file
    if not output_missing(baseline_field_names, test_field_names, "fields"):
        error_found = True

    if error_found:
        sys.exit(1)


# Ensure that field stat names and number of values are same in both files
#
# Pre: field names match
def ensure_field_stats(field_names, baseline_curves, test_curves):
    error_count = 0
    zero_found = False

    # Check stat names match
    for field_name in field_names:
        baseline_stat_names = baseline_curves[field_name].keys()
        test_stat_names = test_curves[field_name].keys()

        # Check if either file has no stats under each field
        if len(baseline_stat_names) == 0:
            print("ERROR: Baseline file had no stats under field '{0}'".format(field_name))
            error_count += 1
            zero_found = True
        if len(test_stat_names) == 0:
            print("ERROR: Test file had no stats under field '{0}'".format(field_name))
            error_count += 1
            zero_found = True

        # if stats are present, check for extra/missing stats in test file
        if not zero_found:
            if not output_missing(baseline_stat_names, test_stat_names, "stats", field_name):
                error_count += 1

        zero_found = False

    if error_count > 0:
        sys.exit(1)

    # Check lengths of all stat values lists
    for field_name in field_names:
        for stat_name in baseline_curves[field_name].keys():
            baseline_values = baseline_curves[field_name][stat_name]
            test_values = test_curves[field_name][stat_name]

            # Check if either file has no stat values under each stat
            if len(test_values) == 0:
                print("ERROR: Baseline file had no stat values under '{0}/{1}'".format(field_name, stat_name))
                error_count += 1
                zero_found = True
            if len(test_values) == 0:
                print("ERROR: Test file had no stat values under '{0}/{1}'".format(field_name, stat_name))
                error_count += 1
                zero_found = True

            # if both have some stat values, make sure they have the same amount
            if not zero_found:
                if len(baseline_values) > len(test_values):
                    print("ERROR: Test file has less entries ({0} vs {1}) than the baseline file under '{2}/{3}'"
                           .format(len(test_values), len(baseline_values), field_name, stat_name))
                    error_count += 1
                elif len(baseline_values) < len(test_values):
                    print("ERROR: Test file has more entries ({0} vs {1}) than the baseline file under '{2}/{3}'"
                          .format(len(test_values), len(baseline_values), field_name, stat_name))
                    error_count += 1

    if error_count > 0:
        sys.exit(1)


# Ensure that field stat values are within tolerance
#
# Pre: field and stat names and lengths are match
def ensure_field_stat_values(field_names, baseline_curves, test_curves, tolerance):
    error_found = False

    # Check if values are within given tolerance
    for field_name in field_names:
        for stat_name in baseline_curves[field_name].keys():
            baseline_values = baseline_curves[field_name][stat_name]
            test_values = test_curves[field_name][stat_name]

            for i in range(len(baseline_values)):
                baseline_value = baseline_values[i]
                test_value = test_values[i]
                if not math.isclose(baseline_value, test_value, abs_tol=tolerance):
                    name = "{0}/{1}[{2}]".format(field_name, stat_name, i)
                    print("ERROR: Test value '{0}' out of tolerance: baseline value={1}, test value={2}"
                          .format(name, baseline_value, test_value))
                    error_found = True

    if error_found:
        sys.exit(1)


def main():
    args = parse_args()

    # Load both files
    with open(args.baseline) as baseline_file:
        baseline_json = json.load(baseline_file)
    with open(args.test) as test_file:
        test_json = json.load(test_file)

    # Start at "curves"
    if not "curves" in baseline_json:
        print("ERROR: Baseline file did not have a 'curves' section")
    baseline_curves = baseline_json["curves"]
    if not "curves" in test_json:
        print("ERROR: Test file did not have a 'curves' section")
    test_curves = test_json["curves"]

    ensure_timesteps(baseline_curves, test_curves)

    # Get both sets of field names
    baseline_field_names = get_field_names(baseline_curves)
    test_field_names = get_field_names(test_curves)
     
    ensure_field_names(baseline_field_names, test_field_names)
    ensure_field_stats(baseline_field_names, baseline_curves, test_curves)
    ensure_field_stat_values(baseline_field_names, baseline_curves, test_curves, args.tolerance)

    print("Success: Test file passed")

if __name__ == "__main__":
    main()
    sys.exit(0)

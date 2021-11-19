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

def ensure_file(path):
    if not os.path.exists(path):
        print("ERROR: Given file does not exist: {0}".format(path))
        sys.exit(1)
    if not os.path.isfile(path):
        print("ERROR: Given file is not a file: {0}".format(path))
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare JSON files")

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


 # filter field names from a list of all possible names
def get_field_names(possible_fields):
    field_names = []
    # known non-field names that should be ignored
    non_field_names = ["t", "sidre_group_name"]
    for key in possible_fields.keys():
        if key in non_field_names:
            continue
        field_names.append(key)
    return field_names

# Return list of items in l1 that are missing from l2
def list_missing(l1, l2):
    missing = [x for x in l1 if x not in l2]
    return missing

# Ensure list of field names are equal and error out with useful message
def ensure_field_names(baseline_field_names, test_field_names):
    error_found = False

    if len(baseline_field_names) == 0:
        print("ERROR: Baseline file had no field names")
        error_found = True
    if len(test_field_names) == 0:
        print("ERROR: Test file had no field names")
        error_found = True

    if error_found:
        sys.exit(1)

    if len(baseline_field_names) > len(test_field_names):
        print("ERROR: Test file is missing field names that are in the baseline file")
        missing = list_missing(baseline_field_names, test_field_names)
        print("       Missing field names: {0}".format(", ".join(missing)))
        error_found = True
    elif len(baseline_field_names) < len(test_field_names):
        print("ERROR: Test file has extra field names not in baseline file")
        missing = list_missing(test_field_names, baseline_field_names)
        print("       Extra field names: {0}".format(", ".join(missing)))
        error_found = True

    if error_found:
        sys.exit(1)


# Ensure that field sample names and number of values are same in both files
def ensure_field_samples(field_names, baseline_curves, test_curves):
    error_count = 0
    zero_found = False

    # Check data types match
    for field_name in field_names:
        baseline_sample_names = baseline_curves[field_name].keys()
        test_sample_names = test_curves[field_name].keys()

        # Check sample names are correct
        if len(baseline_sample_names) == 0:
            print("ERROR: Baseline file had no data types under field name '{0}'".format(field_name))
            error_count += 1
            zero_found = True
        if len(test_sample_names) == 0:
            print("ERROR: Test file had no data types under field name '{0}'".format(field_name))
            error_count += 1
            zero_found = True

        if not zero_found:
            if len(baseline_sample_names) > len(test_sample_names):
                print("ERROR: Test file is missing sample names under field name '{0}'".format(field_name))
                missing = list_missing(baseline_sample_names, test_sample_names)
                print("       Missing sample names: {0}".format(", ".join(missing)))
                error_count += 1
            elif len(baseline_sample_names) < len(test_sample_names):
                print("ERROR: Test field has extra sample names under field name '{0}'".format(field_name))
                missing = list_missing(test_sample_names, baseline_sample_names)
                print("       Extra sample names: {0}".format(", ".join(missing)))
                error_count += 1
        zero_found = False

    if error_count > 0:
        sys.exit(1)

    # Check lengths of all data type value lists
    for field_name in field_names:
        for sample_name in baseline_curves[field_name].keys():
            baseline_values = baseline_curves[field_name][sample_name]
            test_values = test_curves[field_name][sample_name]

            if len(test_values) == 0:
                print("ERROR: Baseline file had no sample values under '{0}/{1}'".format(field_name, sample_name))
                error_count += 1
                zero_found = True
            if len(test_values) == 0:
                print("ERROR: Test file had no sample values under '{0}/{1}'".format(field_name, sample_name))
                error_count += 1
                zero_found = True

            if not zero_found:
                if len(baseline_values) > len(test_values):
                    print("ERROR: Test file has less entries ({0} vs {1}) than the baseline file under '{2}/{3}'"
                           .format(len(test_values), len(baseline_values), field_name, sample_name))
                    error_count += 1
                elif len(baseline_values) < len(test_values):
                    print("ERROR: Test file has more entries ({0} vs {1}) than the baseline file under '{2}/{3}'"
                          .format(len(test_values), len(baseline_values), field_name, sample_name))
                    error_count += 1

    if error_count > 0:
        sys.exit(1)


# Ensure that field sample values are within tolerance
def ensure_field_sample_values(field_names, baseline_curves, test_curves, tolerance):
    error_found = False

    # Check lengths of all data type lists
    for field_name in field_names:
        for sample_name in baseline_curves[field_name].keys():
            baseline_values = baseline_curves[field_name][sample_name]
            test_values = test_curves[field_name][sample_name]

            for i in range(len(baseline_values)):
                baseline_value = baseline_values[i]
                test_value = test_values[i]
                if not math.isclose(baseline_value, test_value, abs_tol=tolerance):
                    name = "{0}/{1}[{2}]".format(field_name, sample_name, i)
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

    # Get both sets of field names
    baseline_field_names = get_field_names(baseline_curves)
    test_field_names = get_field_names(test_curves)
     
    ensure_field_names(baseline_field_names, test_field_names)
    ensure_field_samples(baseline_field_names, baseline_curves, test_curves)
    ensure_field_sample_values(baseline_field_names, baseline_curves, test_curves, args.tolerance)

    print("Success: Test file passed")

if __name__ == "__main__":
    main()
    sys.exit(0)

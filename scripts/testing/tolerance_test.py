import json
import argparse

# ask for file name
file_to_test = input("file name: ") # l2norms last number in last temp listed changed from 9 to 5 (in small_change)
                                    # mins first number in second temp listed changed from 1 to 8 (big_change)
print(file_to_test)

# open & load data from file
with open(file_to_test) as testing_file:
    testing_data = json.load(testing_file)

# open & load original file
with open("/home/zahmed9/serac/scripts/testing/new_output.json") as good_file:
    good_data = json.load(good_file)

# prints if file contains the same data
print(sorted(testing_data.items()) == sorted(good_data.items()))

# python3 verifier.py --json-output output_1.json --reference-output output_2.json  --tolerance 0.0001

"""# compare values in file 
parser = argparse.ArgumentParser(description="Compare JSON files")
parser.add_argument("first_file", type = json, help = "Name of JSON File")
parser.add_argument("second_file", type = json, help = "Name of 2nd JSON File")
parser.add_argument("tolerance", type = double, help = "tolerance amount")
args = parser.parse_args()



def file_checker(jsonfile1, jsonfile2):
    tolerance 
print file_checker(args.first_file,args.second_file)    """
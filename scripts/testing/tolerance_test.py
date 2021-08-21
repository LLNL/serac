import json
import argparse # for command line arguments
import math # for comparing values within tolerance

 # l2norms last number in last temp listed changed from 9 to 8 (in small_change)
 # mins first number in second temp listed changed from 1 to 100 (big_change)
 
 # load name of each field from file & populate them in a list 
def get_field_names(d):    
    field_names = []
    non_field_names = ["t", "sidre_group_name"] # names that should not be included
    for key in d.keys(): # looking @ keys
        if key in non_field_names:
            continue  
        field_names.append(key) # add  names to list
    return field_names

# argparse command line arguments
parser = argparse.ArgumentParser(description="Compare JSON files")
parser.add_argument("-j","--json_output", type= str, required= True, help= "Name of JSON File") 
parser.add_argument("-r","--reference_output", type= str, required= True, help= "Name of 2nd JSON File")
parser.add_argument("-t","--tolerance", type= float, required= True, help= "tolerance amount")

parser.add_argument("-g", "--good_file",action= "store_true", help="Put JSON File")
parser.add_argument("-e","--testing_file",action= "store_true", help="File you want to test")

args = parser.parse_args()

if args.good_file == False: # NOTE: for some reason it isn't passing as true?
    with open(args.json_output) as testing_file: # open & load testing data JSON file
        testing_data = json.load(testing_file)
    if args.testing_file == False:
        with open(args.reference_output) as good_file: # open & load good data JSON file 
            good_data = json.load(good_file)
   # print(sorted(testing_data.items()) == sorted(good_data.items())) # just for testing, change with tolerance

    good_field_names = get_field_names(good_data) # set variable equal to all field names for good data
    testing_field_names = get_field_names(testing_data) # set variable equal to all field names for testing data
 
    # Checker for if the number of field names/keys in testing file is not equal to number field names/keys in good file
  #  if sorted(good_field_names) != sorted(testing_field_names):
    #if len(good_field_names) != len(testing_field_names):
    if len(good_field_names) > len(testing_field_names):
            print("ERROR: testing file doesn't have enough field names")
            exit()
    elif len(good_field_names) < len(testing_field_names):
            print("ERROR: testing field has too many field names:")
            exit()
    elif len(good_field_names) != len(testing_field_names):
            print("ERROR: testing file does not have the same number of field names as the good file.")
            exit()

        
      
    for field_name in good_field_names:   # for each individual field name 
        for data_name in good_data[field_name].keys(): # for dictionaries inside dictionary fields
            good_values = good_data[field_name][data_name] # set good data names from good data
            testing_values = testing_data[field_name][data_name] # set testing data names from testing data
            if len(good_values) != len(testing_values): # if the number of values/indices in each list from the testing file does not match good file, error out
            # NOTE: I wanted to put this closer to the beginning so it could error out earlier but I wasn't able to figure it out (yet)
                print("ERROR: testing file does not have the same values of files as the good file")  
                exit()
                    
        # ALTERNATIVE SOLUTION    
        #    for good_value, testing_value in zip(good_values, testing_values): # iterate over both at once
                #print(good_value, "->", testing_value)
        #        if good_value != testing_value:
        #            print("ERROR: testing file does not have the same number of files as the good file")
        #        if math.isclose(good_value, testing_value, abs_tol = args.tolerance) == False:
        #            print(testing_value, "is not correct")
            
            for value in range(len(good_values)): # iterates through values within the good_values list
                #print(good_values[i], "->", testing_values[i])  # one integer, index on both lists, i.e. testing_values[0] goes with good_values[0]
                if math.isclose(good_values[value], testing_values[value], abs_tol = args.tolerance) == False:
                    print(testing_values[value], "in", field_name, data_name, "is not correct") # print which value from what dictionary field and data name is incorrect from the testing data
                    
           
           
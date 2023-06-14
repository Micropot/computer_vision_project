import yaml

# function to read the YAML file
def YAML_Reader(FILENAME, Parameters):
    print("file currently readed = ", FILENAME)

    with open(FILENAME, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for key, values in data_loaded.items():
            if hasattr(Parameters, key):
                setattr(Parameters, key, values)
            else:
                print("Warning : the variable ", key, " does not exist")
    return Parameters

import yaml

def YAML_Reader(FILENAME, Parameters):
    print("file currently readed = ", FILENAME)

    with open(FILENAME, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for key, values in data_loaded.items():
            #print("key : ", key)
            #print("values : ",values)
            if hasattr(Parameters, key):
                setattr(Parameters, key, values)
            else:
                print("Warning : the variable ", key, " does not exist")
    return Parameters

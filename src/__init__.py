import json

def read_dataset(filename):
    filename_string = '../data/' + filename
    with open(filename_string, 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict

if __name__ == '__main__':
    file_dict = read_dataset('HMWP/hmwp.json')
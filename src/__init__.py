import json

def read_json(filename):
    filename = '../data/' + filename
    with open(filename, 'r') as f:
        json_data = json.load(f)
    return json_data


def read_math23k_json(filename):
    filename = '../data/' + filename
    data_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        count = 0
        string = ''
        for line in f:
            count += 1
            string += line
            if count % 7 == 0:
                data_list.append(json.loads(string))
                string = ''
    return data_list



if __name__ == '__main__':
    file_list = read_json("allArith/questions.json")
    print(len(file_list))
    print(file_list[0]["template"])


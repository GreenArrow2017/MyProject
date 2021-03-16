import json
import random
import os
import tensorflow as tf
from model import *


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


def get_trainingdata_label(filename='allArith/questions.json'):
    file_list = read_json(filename)
    questionList = []
    solutionList = []
    templateList = []
    for file in file_list:
        questionList.append(file['sQuestion'])
        solutionList.append(file['lSolutions'])
        templateList.append(file['template'])
    return questionList, solutionList, templateList


def seperater_training_validate(length=1492):
    indexs = [index for index in range(1492)]
    index_trainings = random.sample(range(0, 1492), int(length * 0.8))
    index_test = list(set(indexs) - set(index_trainings))
    return index_trainings, index_test
    pass


def extract_by_index(indexs, trainings, solutions, templates):
    trainings_new = [trainings[i] for i in indexs]
    solutions_new = [solutions[i] for i in indexs]
    templates_new = [templates[i] for i in indexs]
    return trainings_new, solutions_new, templates_new


def split_train_test(file_name='allArith/questions.json'):
    trainings, solutions, templates = get_trainingdata_label(file_name)
    index_trainings, index_test = seperater_training_validate()
    questions_train, solutions_train, templates_train = extract_by_index(index_trainings, trainings, solutions,
                                                                         templates)
    questions_test, solutions_test, templates_test = extract_by_index(index_test, trainings, solutions,
                                                                      templates)
    return (questions_train, solutions_train, templates_train), \
           (questions_test, solutions_test, templates_test)


def write_data2txt(trainings, templates, mode = 'train'):
    '''
    把问题文本按照类别写入相应文件夹中，如何用数据生成器读取
    :param trainings: 问题文本
    :param templates: 问题模板
    :return: None
    '''
    indexs = [0 for i in range(6)]
    for i in range(len(trainings)):
        q = trainings[i]
        t = templates[i]
        filename = '../data/allArith/' + mode + '/' + str(t) + '/' + str(indexs[int(t)]) + '.txt'
        indexs[int(t)] += 1
        with open(filename, 'w', encoding='utf-8') as w:
            w.write(q)
    pass


class data_generator():
    def __init__(self):
        self.train_generator = tf.keras.preprocessing.text_dataset_from_directory(
            '../data/allArith/train',
            labels='inferred',
            label_mode='categorical',
            class_names=None,
            batch_size=32,
            max_length=None,
            shuffle=True,
            seed=1337,
            validation_split=0.2,
            subset='training',
        )

        self.valid_generator = tf.keras.preprocessing.text_dataset_from_directory(
            '../data/allArith/train',
            labels='inferred',
            label_mode='categorical',
            class_names=None,
            batch_size=32,
            max_length=None,
            shuffle=True,
            seed=1337,
            validation_split=0.2,
            subset='validation',
        )

        self.test_generator = tf.keras.preprocessing.text_dataset_from_directory(
            '../data/allArith/test',
            labels='inferred',
            label_mode='categorical',
            batch_size=32,
        )

    def get_generator(self):
        return self.train_generator, self.valid_generator, self.test_generator



if __name__ == '__main__':
    # questions, solutions, templates = get_trainingdata_label()
    # (questions_train, solutions_train, templates_train), (
    # questions_test, solutions_test, templates_test) = split_train_test()
    # write_data2txt(questions_train, templates_train, 'train')
    # write_data2txt(questions_test, templates_test, 'test')
    generator = data_generator()
    train_ge, valid_ge, test_ge = generator.get_generator()

    # model = model().build_model_FastText()
    # model.fit(train_ge, validation_data=valid_ge, epochs=5)
    # model.evaluate(test_ge)
    #
    # model.save('../model/FastText', save_format='tf')



    loaded_model = tf.keras.models.load_model('../model/FastText')
    x = 'Joan has 9.0 blue balloons but gained 2.0 more.  How many blue balloons does Joan have now?'
    print(np.argmax(loaded_model.predict([x])))




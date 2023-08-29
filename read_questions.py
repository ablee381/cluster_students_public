import zipfile
import re
import os
import dill
import matplotlib.pyplot as plt


def pull_tex_from_zip(zip_name):
    file_name = None
    z = zipfile.ZipFile(zip_name)
    file_list = z.namelist()
    for name in file_list:
        if name[-4:] == '.tex':
            file_name = name
            break
    assert (file_name is not None)
    f = z.open(file_name, 'r')
    file_str = f.read()
    f.close()
    z.close()
    return file_str.decode('utf8')


def find_answer_choices(doc_parts):
    """Tries to find where all the answer choices are. Not robust because the line must start with the answer choice"""
    out = []
    for i in range(len(doc_parts) - 2):
        if doc_parts[i] != '' and doc_parts[i + 1] != '':
            if [doc_parts[i][0], doc_parts[i + 1][0]] == ['A', 'B']:
                start_ind = i
                MC = 'CDEFGH'[::-1]
                j = i + 2
                while (len(MC) > 0 and len(doc_parts[j]) > 0) and doc_parts[j][0] == MC[-1]:
                    MC = MC[:-1]
                    j += 1
                end_ind = j
                out.append((start_ind, end_ind))
    return out


def find_question_starts(doc_parts):
    """Assumes the test questions are numbers 1,2,3,..."""
    out = []
    question_pointer = 1
    for i in range(len(doc_parts)):
        x = re.search(str(question_pointer), doc_parts[i])
        if x:
            # Do a secondary search to make sure there are no other numbers in this line
            y = re.search(r'\d', doc_parts[i][:x.span()[0]] + doc_parts[i][x.span()[1]:])
            if not y:
                out.append(i)
                question_pointer += 1
    return out
def find_next_line(doc_parts, pointer):
    """Pointer is pointing to a line with stuff. Find the next line with stuff"""
    pointer += 1
    line = doc_parts[pointer]
    while len(line) == 0:
        pointer += 1
        line = doc_parts[pointer]
    return pointer


def pull_test_w_answers(tex_str):
    doc_parts = re.split('\n', tex_str)
    answer_ind_list = find_answer_choices(doc_parts)
    out_list = []
    for i in range(len(answer_ind_list)):
        answer_ind = answer_ind_list[i]
        if i == 0:
            pointer = answer_ind[0] - 1
            x = re.search('[Qq]uestion', doc_parts[pointer])
            while not x and pointer > 0:
                pointer -= 1
                x = re.search('[Qq]uestion', doc_parts[pointer])
            question_text = '\n'.join(doc_parts[pointer:answer_ind[1]]).strip()
            out_list.append(question_text)
        else:
            question_parts = doc_parts[answer_ind_list[i - 1][1] + 1:answer_ind[1]]
            question_text = '\n'.join(question_parts).strip()
            out_list.append(question_text)
    return out_list


def pull_test_w_questions(tex_str):
    doc_parts = re.split('\n', tex_str)
    question_ind_list = find_question_starts(doc_parts)
    out_list = []
    for i in range(len(question_ind_list)):
        if i < len(question_ind_list)-1:
            start = question_ind_list[i]
            end = question_ind_list[i+1]
            question_parts = doc_parts[start:end]

        else:
            question_parts = doc_parts[question_ind_list[i]:]
        question_text = '\n'.join(question_parts).strip()
        out_list.append(question_text)
    return out_list


def training_data_main(out_name):
    dir_name = 'tex_zips'
    zip_list = os.listdir(dir_name)
    data_dict = {'question_text': [], 'standard': []}
    for zip_name in zip_list:
        standard = zip_name[:-8]
        print(standard)
        file_str = pull_tex_from_zip(dir_name + '/' + zip_name)
        question_list = pull_test(file_str)
        data_dict['question_text'] += question_list
        data_dict['standard'] += [standard] * len(question_list)
    with open(out_name+'.pkd', 'wb') as f:
        dill.dump(data_dict, f)
    return data_dict


def main(zip_name):
    file_str = pull_tex_from_zip(zip_name)
    question_list = pull_test_w_questions(file_str)
    for i in range(6):
        print('QUESTION', i+1)
        print(question_list[i])


if __name__ == '__main__':
    main('demo/unit_test.tex.zip')
    # file_handle = 'training_data'
    # file_name = file_handle + '.pkd'
    # if file_name in os.listdir():
    #     with open(file_name, 'rb') as f:
    #         data_dict = dill.load(f)
    # else:
    #     data_dict = main(file_handle)
    # standards = data_dict['standard']
    # item_counter = Counter(standards)
    # item_count = [item_counter[key]/len(standards) for key in item_counter]
    # plt.hist(item_count)
    # plt.xlabel('Percent Representation of training data')
    # plt.ylabel('Number of standards')
    # plt.savefig('algebraI_training_data.png')


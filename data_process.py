import numpy as np
import os
import pickle

NE_code = [ '01_01' , '02_01' , '03_01' , '04_01' , '04_02']
Other_lightness = ['00','01','02','03','04','05','06','07','08','10','11','12','13','14','15','16','17','18','19']
def construct_pairs(train_persons, lms):

    train_pairs = []

    for tp in train_persons:
        tp_imgs = []
        for lm in lms:
            if lm.split("_")[0] == tp:
                tp_imgs.append(lm)

        frontal = []
        p15 = []
        n15 = []

        for tpi in tp_imgs:
            pose = tpi.split("_")[3]

            if pose == '051':
                frontal.append(tpi)
            elif pose == '050':
                p15.append(tpi)
            else:
                n15.append(tpi)

        # print('frontal', frontal)
        # print('p15', p15)
        # print('n15', n15)

        if len(frontal) > 1:
            basic = 0
            for fl in frontal:
                # 中性表情 01_01 , 02_01 , 03_01 , 04_01 , 04_02
                if fl[4:9] in NE_code:
                    basic = fl
                    break

            if not basic == 0:
                for fl in frontal:
                    if fl == basic:
                        continue
                    train_pairs.append([basic, fl])

        if len(p15) > 1:

            basic = 0
            for fl in p15:
                # 中性表情 01_01 , 02_01 , 03_01 , 04_01 , 04_02
                if fl[4:9] in NE_code:
                    basic = fl
                    break

            if not basic == 0:
                for fl in p15:
                    if fl == basic:
                        continue
                    train_pairs.append([basic, fl])

        if len(n15) > 1:

            basic = 0
            for fl in n15:
                # 中性表情 01_01 , 02_01 , 03_01 , 04_01 , 04_02
                if fl[4:9] in NE_code:
                    basic = fl
                    break

            if not basic == 0:
                for fl in n15:
                    if fl == basic:
                        continue
                    train_pairs.append([basic, fl])

    return  train_pairs

def construct_pairs_dense(train_persons, lms, other_lightness_lms):
    train_pairs = []

    for tp in train_persons:
        print('tp',tp)
        tp_imgs = []
        for lm in lms:
            if lm.split("_")[0] == tp:
                tp_imgs.append(lm)

        frontal = []
        p15 = []
        n15 = []

        for tpi in tp_imgs:
            pose = tpi.split("_")[3]

            if pose == '051':
                frontal.append(tpi)
            elif pose == '050':
                p15.append(tpi)
            else:
                n15.append(tpi)

        # print('frontal', frontal)
        # print('p15', p15)
        # print('n15', n15)

        if len(frontal) > 1:
            for fl_1 in frontal:
                for fl_2 in frontal:
                    if fl_1 == fl_2:
                        continue
                    train_pairs.append([fl_1, fl_2])
                    base_1 = fl_1[:-6]
                    base_2 = fl_2[:-6]
                    for light in Other_lightness:
                        other_1 = base_1+light+'.txt'
                        other_2 = base_2+light+'.txt'
                        if (other_1 in other_lightness_lms) and (other_2 in other_lightness_lms):
                            train_pairs.append([other_1, other_2])

        if len(p15) > 1:
            for fl_1 in p15:
                for fl_2 in p15:
                    if fl_1 == fl_2:
                        continue
                    train_pairs.append([fl_1, fl_2])
                    base_1 = fl_1[:-6]
                    base_2 = fl_2[:-6]
                    for light in Other_lightness:
                        other_1 = base_1 + light + '.txt'
                        other_2 = base_2 + light + '.txt'
                        if (other_1 in other_lightness_lms) and (other_2 in other_lightness_lms):
                            train_pairs.append([other_1, other_2])

        if len(n15) > 1:
            for fl_1 in n15:
                for fl_2 in n15:
                    if fl_1 == fl_2:
                        continue
                    train_pairs.append([fl_1, fl_2])
                    base_1 = fl_1[:-6]
                    base_2 = fl_2[:-6]
                    for light in Other_lightness:
                        other_1 = base_1 + light + '.txt'
                        other_2 = base_2 + light + '.txt'
                        if (other_1 in other_lightness_lms) and (other_2 in other_lightness_lms):
                            train_pairs.append([other_1, other_2])

    return train_pairs



def constract_trainset():
    lm_path = r'D:\workspace\common_database\multipie_processed\multipie_128_lm'
    other_lightness_lm_path = r'D:\workspace\common_database\multipie_processed\multipie_128_lm'

    # lms里面的都是09光照的
    lms = os.listdir(lm_path)

    new_lms = []

    for lm in lms:
        if lm[-6:-4] == '09':
            new_lms.append(lm)

    lms = new_lms


    other_lightness_lms = os.listdir(other_lightness_lm_path)
    persons = []

    for lm in lms:
        persons.append(lm.split("_")[0])

    persons = list(set(persons))
    persons = sorted(persons, key=lambda x: int(x))

    print('train')
    person_num = len(persons)
    train_person = persons[0:int(0.9*person_num)]
    for p in train_person:
        print(p)
    print('test')
    test_person = persons[int(0.9*person_num):]
    for p in test_person:
        print(p)


    # train_persons = persons[:-10]
    # test_persons = persons[-10:]

    # train_pairs = construct_pairs_dense(train_persons, lms, other_lightness_lms)
    #
    # test_pairs = construct_pairs_dense(test_persons, lms, other_lightness_lms)
    #
    # with open('train_pairs.pickle', 'wb') as f:
    #     pickle.dump(train_pairs, f, protocol=-1)
    #
    # with open('test_pairs.pickle', 'wb') as f:
    #     pickle.dump(test_pairs, f, protocol=-1)
    #
    # print('train_pairs', train_pairs)
    # print(len(train_pairs))


if __name__ == '__main__':
    constract_trainset()

    # lm_img = np.zeros((3, 2, 2, 3)) - 1
    # add = np.zeros((3, 2, 2, 1))
    # print('lm_img',np.concatenate([lm_img,add], axis=3))

    # lm = np.asarray([['2','3'],['4','5']])[[1,0]]
    # print(lm)
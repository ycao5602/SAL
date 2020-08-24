import os
import numpy as np
import scipy.io as sio
from collections import Counter


def import_peta(dataset_dir):
    peta_dir = os.path.join(dataset_dir,'PETA')
    if not os.path.exists(peta_dir):
        print('Please Download PETA Dataset and check if the sub-folder name exists')

    file_list = os.listdir(peta_dir)

    for name in file_list:
        id = name.split('.')[0]
        globals()[id] = name

    f = sio.loadmat(os.path.join(dataset_dir, 'PETA.mat'))
    attr_dict = {}
    nrow, ncol = f['peta'][0][0][0].shape
    sem_dict={}
    sem_list=[]
    for id in range(nrow):
        attr_dict[str(id + 1).zfill(5)] = f['peta'][0][0][0][id][4:]
        sem_id = str(int("".join(map(str,attr_dict[str(id + 1).zfill(5)])), base=2))
        if sem_id not in sem_list:
            sem_dict[sem_id] = [str(id + 1).zfill(5)]
            sem_list.append(sem_id)
        else:
            sem_dict[sem_id].append(str(id + 1).zfill(5))
    attributes = []  # gives the names of attributes, label
    for i in range(ncol-4):
        attributes.append(f['peta'][0][0][1][i][0][0])

    # Already know that there are 7769 semantic ids. Randomly pick 6769 for training and 1000 for testing.
    new_sem_dict = {}
    for sem_id in sem_dict.keys():
        id_list = sem_dict[sem_id]
        if not len(id_list)==1:
            new_sem_dict[sem_id]=id_list
    sem_dict = new_sem_dict.copy()
    sem_list = list(sem_dict.keys())
    np.random.seed(1)
    num_sem = len(sem_dict.keys())
    training_sems = np.random.choice(num_sem, num_sem-200, replace=False)

    globals()['train'] = {}
    globals()['query'] = {}
    globals()['gallery'] = {}
    globals()['train']['data'] = []
    globals()['train']['ids'] = []
    globals()['query']['data'] = []
    globals()['query']['ids'] = []
    globals()['gallery']['data'] = []
    globals()['gallery']['ids'] = []
    train_attribute = {}
    test_attribute = {}
    for sem_id in sem_dict.keys():
        id_list = sem_dict[sem_id]
        # set a same camid for all images. Camid is not used in this project.
        camid = np.int64(1)
        if sem_list.index(sem_id) in training_sems:
            for id in id_list:
                name = globals()[id]
                images = os.path.join(peta_dir,name)
                globals()['train']['ids'].append(id)
                globals()['train']['data'].append([images, np.int64(globals()['train']['ids'].index(id)), id, camid, name])
                train_attribute[id] = attr_dict[id]
        else:
            for id in id_list:
                name = globals()[id]
                images = os.path.join(peta_dir,name)
                globals()['query']['ids'].append(id)
                globals()['gallery']['ids'].append(id)
                globals()['query']['data'].append([images, np.int64(globals()['query']['ids'].index(id)), id, camid, name])
                globals()['gallery']['data'].append([images, np.int64(globals()['gallery']['ids'].index(id)), id, camid, name])
                test_attribute[id] = attr_dict[id]
    return train, query, gallery, train_attribute, test_attribute, attributes


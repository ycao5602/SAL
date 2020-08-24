import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from .reid_dataset import import_MarketDuke_nodistractors
from .reid_dataset import import_Market1501Attribute_binary
from .reid_dataset import import_peta
import os.path as osp
from collections import Counter



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class Base_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, num_train, seed):
        self.dataset_name = dataset_name
        if dataset_name == 'Market-1501':
            self.train, self.query, self.gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)
            self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'PETA':
            self.train, self.query, self.gallery, self.train_attr, self.test_attr, self.label = import_peta(data_dir)
        else:
            print('Input should only be Market1501 or PETA')
        # print('labels: ',self.label)
        semantic_dict = {}  # key is decimal semantic id, value is number of people belong to this id
        num_count = -1
        self.sem_dict = {}  # key is person id, value is the corresponding semantic id
        self.sem_order = {}  # key is decimal semantic id, value is the order of occurrence.
        # Just used to assign some simple number for simplicity and to avoid numerical instability.

        a = np.zeros(len(self.label))
        for k, v in self.train_attr.items():
            '''
            k is person id and v is the label vector.
            map an integer array to string and join the elements to become one single integer.
            finally change from binary to decimal.
            '''
            a += np.array(v)
            semantic_id = str(int("".join(map(str, v)), base=2))
            if semantic_id in semantic_dict:
                semantic_dict[semantic_id] += 1
            else:
                semantic_dict[semantic_id] = 1
                num_count += 1
                self.sem_order[semantic_id] = num_count

            self.sem_dict[k] = semantic_id

        if num_train is None or num_train > len(self.train['data']):
            num_train = len(self.train['data'])
            print('num_train',num_train)

        # np.random.seed(seed)
        # if num_train < len(self.train['data']):
        #     selected_data = list(np.random.choice(range(len(self.train['data'])), num_train, replace=False))
        #
        #     train_temp = {'ids':[],'data':[]}
        #
        #
        #     for i in range(len(self.train['data'])):
        #         data = self.train['data'][i]
        #         if i in selected_data:
        #             train_temp['data'].append(data)
        #             if data[2] not in train_temp['ids']:
        #                 train_temp['ids'].append(data[2])
        #         else:
        #             self.unannotated['data'].append(data)
        #             if data[2] not in self.unannotated['ids']:
        #                 self.unannotated['ids'].append(data[2])
        #
        #
        #
        #
        #     self.train = train_temp.copy()

        self.w = a / len(self.train_attr.keys())
        self.average = a.sum()/len(self.train_attr.items())
        # print('w values：', self.w)
        self.num_sems = sum(Counter(semantic_dict.values()).values())

        semantic_dict_test = {}
        self.sem_order_test = {}
        self.sem_dict_test = {}

        num_count = -1
        for k, v in self.test_attr.items():
            semantic_id = str(int("".join(map(str, v)), base=2))

            if semantic_id in semantic_dict_test:
                semantic_dict_test[semantic_id] += 1
            else:
                semantic_dict_test[semantic_id] = 1
                num_count+=1
                self.sem_order_test[semantic_id] = num_count

            self.sem_dict_test[k] = semantic_id

        self.test_sem_num = len(semantic_dict_test.keys())
        self.test_num = len(self.gallery['data']+self.query['data'])
        self.total_sem = len(list(set(list(semantic_dict_test.keys())+list(semantic_dict.keys()))))

        sid_list=[]
        pid_list=[]
        new_query = {}
        new_query['ids'] = []
        new_query['data'] = []
        for i in range(len(self.query['ids'])):
            if self.sem_dict_test[self.query['ids'][i]] not in sid_list:
                sid_list.append(self.sem_dict_test[self.query['ids'][i]])
                pid_list.append(self.query['ids'][i])
                new_query['ids'].append(self.query['ids'][i])
                # new_query['data'].append(self.gallery['data'][i])

        # print('number of query sid: ',len(sid_list),len(pid_list))

        for data in self.query['data']:
            if len(pid_list)==0:
                break
            if str(data[2]) == str(pid_list[0]):
                new_query['data'].append(data)
                pid_list.pop(0)
        self.query_temp = self.query # keep all query images
        self.gallery['data'] = list(reversed(self.gallery['data']))
        self.gallery['ids'] = list(reversed(self.gallery['ids']))
        self.query = new_query.copy()
        # print('folder query data: ',len(self.query['data']))

        # query_sem = []
        # for data in self.query['data']:
        #     id = data[2]
        #     if not self.sem_dict_test[id] in query_sem:
        #         query_sem.append(self.sem_dict_test[id])
        self.num_query_sem = len(self.query['ids'])

class Train_Dataset():
    def __init__(self, base, transforms=None, train_val='train' ):
        train, query, gallery = base.train, base.query, base.gallery
        train_attr, test_attr, self.label = base.train_attr, base.test_attr, base.label
        self.sem_order = base.sem_order
        self.sem_dict = base.sem_dict
        self.num_ids = len(train['ids'])
        self.num_labels = len(self.label)
        self.num_images = len(train['data'])
        self.num_query = len(query['data'])
        cam=[]
        for i in range(self.num_images):
            cam.append(train['data'][i][3])
        self.num_cam=len(list(set(cam)))

        '''
        considering changing the train_val to train only. Doesn't seem to be useful with no validation set in the dataset.
        '''

        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
            self.train_attr = train_attr
        elif train_val == 'query':
            self.train_data = query['data']
            self.train_ids = query['ids']
            self.train_attr = test_attr
        elif train_val == 'gallery':
            self.train_data = gallery['data']
            self.train_ids = gallery['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or query or gallery')

        self.num_ids = len(self.train_ids)

        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        '''
        return one image at a time
        '''
        img_path = self.train_data[index][0]

        i = self.train_data[index][1]
        # i is the order of this person in all identities (in training data)

        id = self.train_data[index][2]

        cam = self.train_data[index][3]

        label = np.asarray(self.train_attr[id])
        label = np.int32(label)

        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)

        name = self.train_data[index][4]

        sem=self.sem_order[self.sem_dict[id]]

        return data, i, label, id, cam, name, sem

    def __len__(self):
        return len(self.train_data)

    def num_label(self):
        return self.num_labels


    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label


class Test_Dataset():
    def __init__(self, base, transforms=None, query_gallery='gallery' ):

        train, query, gallery, query_temp = base.train, base.query, base.gallery, base.query_temp
        self.train_attr, self.test_attr, self.label = base.train_attr, base.test_attr, base.label
        self.sem_order_test = base.sem_order_test
        self.sem_dict_test = base.sem_dict_test
        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        elif query_gallery == 'all':
            if base.dataset_name == 'PETA' :
                self.test_data = gallery['data']
            else:
                self.test_data = gallery['data'] + query_temp['data']
            self.test_ids = gallery['ids']
        else:
            print('Input should only be query or gallery;')

        self.test_num = len(self.test_data)
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        # Returns the statistics of one image. (for testing)

        img_path = self.test_data[index][0]
        id = self.test_data[index][2]
        label = np.asarray(self.test_attr[id])
        label = np.int32(label)
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        name = self.test_data[index][4]
        sem = self.sem_dict_test[id]
        sem = self.sem_order_test[sem]
        camid = self.test_data[index][3]
        camid = str(camid)+'_'+str(id)
        return data, label, id, camid, img_path, sem

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label


from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader
import os
#from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .transforms import build_transforms
from .samplers import RandomIdentitySampler
from .folder import Train_Dataset, Test_Dataset, Base_Dataset
from math import ceil

class BaseDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    @property
    def num_train_sems(self):
        return self._num_train_sems

    @property
    def num_train_attrs(self):
        return self._num_train_attributes


    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']

    '''
    THIS IS DESIGNED FOR VISUALISING THE DATA. NOT IMPLEMENTING AT THIS MOMENT.
    -----------------------------------------------------------------------------   
    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']
    '''

class ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 num_train,
                 seed,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=128,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 num_instances=4, # number of instances per identity (for RandomIdentitySampler)
                 ):
        super(ImageDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.pin_memory = True if self.use_gpu else False

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True)
        transform_test = build_transforms(self.height, self.width, is_train=False)

        print("=> Initializing TRAIN (source) datasets")

        self._num_train_pids = 0
        self._num_train_cams = 0

        name = self.source_names[0]
        base_dataset = Base_Dataset(self.root, dataset_name=name, num_train=num_train, seed=seed)
        train_dataset = Train_Dataset(base_dataset, train_val='train')
        self._num_train_pids = train_dataset.num_ids
        self._num_train_cams = train_dataset.num_cam
        self._num_train_images = train_dataset.num_images
        self._num_train_attributes = train_dataset.num_labels
        self._num_train_sems=base_dataset.num_sems
        self.total_sem = base_dataset.total_sem
        self.label = base_dataset.label
        self.w = base_dataset.w
        self.average = base_dataset.average
        self.trainloader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )



        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        
        for name in self.target_names:

            test_dataset = Test_Dataset(base_dataset, query_gallery='all')
            query_dataset = Test_Dataset(base_dataset, query_gallery='query')
            self.testloader_dict[name]['query'] = DataLoader(
                query_dataset,
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                test_dataset,
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )
            self._num_test_images = test_dataset.test_num
            self._num_test_sems = base_dataset.test_sem_num
            self._num_query_images = train_dataset.num_query
            self._num_query_sems = base_dataset.num_query_sem

            '''
            THIS IS DESIGNED FOR VISUALISING THE DATA.
            -----------------------------------------------------------------------------   
            '''
            self.testdataset_dict[name]['query'] = query_dataset
            self.testdataset_dict[name]['gallery'] = test_dataset


        print("\n")
        print("  **************** Summary ****************")
        print("  dataset name           : {}".format(self.source_names))
        print("  # train images        : {}".format(self._num_train_images))
        print("  # train attributes    : {}".format(self._num_train_attributes))
        print("  # train categories    : {}".format(self._num_train_sems))
        print("  # gallery images      : {}".format(self._num_test_images))
        print("  # query/gallery categories    : {}".format(self._num_query_images))
        print("  categories   in total : {}".format(self.total_sem))
        print("  # batch size          : {}".format(self.train_batch_size))
        print("  *****************************************")
        print("\n")

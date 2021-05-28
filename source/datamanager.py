import os, random

import numpy as np
import source.utils as utils

class DataSet(object):

    def __init__(self, dir='dataset'):

        print("\n** Prepare the Dataset")

        _, self.data_tr = utils.read_pickle(path=os.path.join(dir, 'train.cpkl.gz'))
        self.data_val = self.data_tr[-35:]
        self.data_tr = self.data_tr[:-35]
        _, self.data_te = utils.read_pickle(path=os.path.join(dir, 'test.cpkl.gz'))
        self.shuffle_training()
        self.reset_index()

        try: self.num_class = len(list(set(list(self.data_tr[0]['label'][:, -1]))))
        except: self.num_class = len(list(set(list(self.data_te[0]['label'][:, -1]))))

        self.num_tr, self.num_te, self.num_val = \
            len(self.data_tr), len(self.data_te), len(self.data_val)

        minibatch, terminate = self.next_batch(batch_size=1, ttv=0)
        self.dim_node_feat = minibatch['r_vertex'].shape[-1]
        self.dim_edge_near = minibatch['r_edge'].shape[-2]
        self.dim_edge_feat = minibatch['r_edge'].shape[-1]

        self.reset_index()
        print("\n* Summary")
        print("Training   : %d" %(self.num_tr))
        print("Validation : %d" %(self.num_val))
        print("Test       : %d" %(self.num_te))

    def reset_index(self):

        self.idx_tr, self.idx_te, self.idx_val = 0, 0, 0
        self.inter_tr, self.inter_te, self.inter_val = 0, 0, 0

    def shuffle_training(self):

        random.shuffle(self.data_tr)
        for idx, _ in enumerate(self.data_tr):
            np.random.shuffle(self.data_tr[idx]['label'])

    def next_batch(self, batch_size=1, ttv=0):

        if(ttv == 0):
            idx_d, inter, data = self.idx_tr, self.inter_tr, self.data_tr
        elif(ttv == 1):
            idx_d, inter, data = self.idx_te, self.inter_te, self.data_te
        else:
            idx_d, inter, data = self.idx_val, self.inter_val, self.data_val

        batch, terminate = {}, False

        while(True):
            try:
                data_protein = data[idx_d]
            except:
                idx_d = 0
                inter = 0
                terminate = True
                self.shuffle_training()
                try: del data_protein, pairs
                except: pass
                break
            else:
                pairs = data_protein['label'][inter:inter+batch_size]
                inter = inter + batch_size
                if(inter >= data_protein['label'].shape[0]):
                    idx_d += 1
                    inter = 0
                break

        if(ttv == 0): self.idx_tr, self.inter_tr = idx_d, inter
        elif(ttv == 1): self.idx_te, self.inter_te = idx_d, inter
        else: self.idx_val, self.inter_val = idx_d, inter

        try:
            for name_key in list(data_protein.keys()):
                if(name_key == 'label'):
                    batch['label'] = pairs.astype(np.int32)
                    one_hots = []
                    for idx_p, pair in enumerate(batch['label']):
                        if(pair[-1] == -1): one_hots.append(np.diag(np.ones(self.num_class))[0])
                        else: one_hots.append(np.diag(np.ones(self.num_class))[1])
                    batch['label_1hot'] = np.asarray(one_hots).astype(np.float32)
                else:
                    if('hood' in name_key): batch[name_key] = data_protein[name_key].astype(np.int32)
                    else:
                        try: batch[name_key] = data_protein[name_key].astype(np.float32)
                        except: batch[name_key] = data_protein[name_key]
        except: pass

        return batch, terminate

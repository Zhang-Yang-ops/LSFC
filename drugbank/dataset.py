import os
import torch
import codecs
import pickle
import numpy as np
import pandas as pd
from subword_nmt.apply_bpe import BPE
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))


def drug2emb_encoder(x):
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph):
        self.data_df = data_df
        self.drug_graph = drug_graph

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        smiles_h = []
        smiles_t = []
        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_SMILES'], row['Drug2_SMILES'], row['Y'], row['Neg samples']
            smiles_h.append(Drug1_ID)
            smiles_t.append(Drug2_ID)
            Neg_ID, Ntype = Neg_samples.split('$')
            h_graph = torch.from_numpy(self.drug_graph.get(Drug1_ID))
            t_graph = torch.from_numpy(self.drug_graph.get(Drug2_ID))
            n_graph = torch.from_numpy(self.drug_graph.get(Neg_ID))

            pos_pair_h = h_graph
            pos_pair_t = t_graph

            if Ntype == 'h':
                smiles_h.append(Neg_ID)
                smiles_t.append(Drug2_ID)
                neg_pair_h = n_graph
                neg_pair_t = t_graph
            else:
                smiles_h.append(Drug1_ID)
                smiles_t.append(Neg_ID)
                neg_pair_h = h_graph
                neg_pair_t = n_graph

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = torch.stack(head_list, dim=0)
        tail_pairs = torch.stack(tail_list, dim=0)
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)
        d_v = []
        p_v = []
        input_mask_d = []
        input_mask_p = []
        for i in range(len(smiles_h)):
            d_v1, input_mask_d1 = drug2emb_encoder(smiles_h[i])
            p_v1, input_mask_p1 = drug2emb_encoder(smiles_t[i])
            d_v.append(torch.from_numpy(d_v1))
            p_v.append(torch.from_numpy(p_v1))
            input_mask_d.append(torch.from_numpy(input_mask_d1))
            input_mask_p.append(torch.from_numpy(input_mask_p1))

        d_v = torch.stack(d_v, dim=0)
        p_v = torch.stack(p_v, dim=0)
        input_mask_d = torch.stack(input_mask_d, dim=0)
        input_mask_p = torch.stack(input_mask_p, dim=0)
        return head_pairs, tail_pairs, rel, label, d_v, p_v, input_mask_d, input_mask_p


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.2):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['Y'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df


def load_ddi_dataset(root, batch_size, fold=0):
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
    train_df, val_df = split_train_valid(train_df, fold=fold)

    train_set = DrugDataset(train_df, drug_graph)
    val_set = DrugDataset(val_df, drug_graph)
    test_set = DrugDataset(test_df, drug_graph) 
    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)
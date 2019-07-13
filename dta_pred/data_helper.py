## SOME OF THE METHODS AND CLASSES ARE TAKEN FROM github.com/hkmztrk/deepdta

import json
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
from os import path
import sys
import glob
from sklearn.model_selection import KFold, PredefinedSplit, train_test_split
import urllib.request
import re
from tqdm import tqdm

## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "U": 19, "T": 20, "W": 21,
                "V": 22, "Y": 23, "X": 24,
                "Z": 25 }

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

def get_kinase_domains(dataset):
    """

    :param dataset: Dataframe consisting of UniProt_Id column.
    :return: dataset with an extra `fasta` column which contains kinase domain of the protein.
    """
    target_ids = dataset['UniProt_Id'].unique()
    dataset.loc[:, 'fasta'] = ''

    for target_id in tqdm(target_ids):
        url = 'https://www.uniprot.org/uniprot/' + target_id + '.txt'
        raw_data = urllib.request.urlopen(url).read().decode("utf-8")
        kinase_domain = re.search(r"\nFT([\t ]+)DOMAIN([\t ]+)([0-9]+)([\t ]+)([0-9]+)([\t ]+)Protein kinase", raw_data)
        if kinase_domain is not None:
            kinase_domain = kinase_domain.groups()
            kinase_domain = kinase_domain[2], kinase_domain[4]

        raw_blast = ""
        i = 1
        while (raw_blast == "" and i < 5):
            if (i > 1):
                time.sleep(0.2)

            url = 'https://www.uniprot.org/blast/?about=' + target_id

            if kinase_domain is not None:
                url += '[' + kinase_domain[0] + '-' + kinase_domain[1] + ']&key=Domain'

            raw_blast = urllib.request.urlopen(url).read().decode("utf-8")
            i = i + 1

        fasta_sequence = ''.join(
            re.search(r'sequence-textarea">(.*?)\n(([^<]*?))</textarea>', raw_blast).groups()[1]).replace('\n', '')
        if kinase_domain is not None:
            assert len(fasta_sequence) == int(kinase_domain[1]) - int(kinase_domain[0]) + 1

        dataset.loc[dataset['UniProt_Id'] == target_id, 'fasta'] = fasta_sequence

    return dataset

## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

def label_smiles(smiles_string, MAX_SMI_LEN, smiles_to_integer):
    """
    Convert SMILES string into a vector of integers

    :param smiles_string: SMILES string
    :param MAX_SMI_LEN: length of the encoded vector. If SMILES string length is less than this, zero padding will
    be applied, if it is greater, then string will be clipped.
    :param smiles_to_integer: A dictionary to encode each letter in the SMILES string to an integer value
    :return: np.ndarray, encoded SMILES
    """
    if(MAX_SMI_LEN == None):
        X = np.zeros(len(smiles_string))
    else:
        X = np.zeros(MAX_SMI_LEN)

    for i, ch in enumerate(smiles_string[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
        if ch not in smiles_to_integer:
            print('Not found', ch, 'at the index:', i, 'total:', len(smiles_string))
        else:
            X[i] = smiles_to_integer[ch]

    return X

def label_sequence(line, MAX_SEQ_LEN, aminoacid_to_integer):
    """
    Convert aminoacid sequence into a vector of integers

    :param line: Aminoacid sequence
    :param MAX_SEQ_LEN: length of the encoded vector. If aminoacid sequence length is less than this, zero padding will
    be applied, if it is greater, then string will be clipped.
    :param aminoacid_to_integer: A dictionary to encode each letter in the SMILES string to an integer value
    :return: np.ndarray, encoded aminoacid sequence
    """

    if MAX_SEQ_LEN == None:
        X = np.zeros(len(line))
    else:
        X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = aminoacid_to_integer[ch]

    return X #.tolist()

class DataSet(object):
    """Helper class to parse drug target interaction datasets"""

    def __init__(self, dataset_path, dataset_name, seqlen, smilen, protein_format='sequence', drug_format='labeled_smiles',
                 mol2vec_model_path=None, mol2vec_radius=1, biovec_model_path=None):
        """

        :param dataset_path:
        :param dataset_name:
        :param seqlen:
        :param smilen:
        :param protein_format:
        :param drug_format:
        :param mol2vec_model_path:
        :param mol2vec_radius:
        :param biovec_model_path:
        """
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        #self.NCLASSES = n_classes
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN
        self.dataset_path = dataset_path
        self.fpath = path.join(dataset_path, dataset_name)
        self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        self.protein_format = protein_format
        self.drug_format = drug_format

        self.mol2vec_model_path = mol2vec_model_path
        self.mol2vec_radius = mol2vec_radius
        self.biovec_model_path = biovec_model_path
        with open(path.join(self.fpath, 'type.txt'), 'r') as f:
            self.interaction_type = f.read()

        if path.exists(path.join(self.fpath, 'proteins.txt')):
            self.dataset_type = 'txt'
        else:
            self.dataset_type = 'csv'

    def parse_data(self, with_label=True):
        if self.dataset_type == 'txt':
            return self.parse_txt()
        else:
            return self.parse_csv(with_label=with_label)

    def parse_txt(self):
        fpath = self.fpath

        ligands = json.load(open(path.join(fpath, "ligands_iso.txt")), object_pairs_hook=OrderedDict)

        if self.protein_format == 'sequence' or self.protein_format == 'biovec':
            proteins = json.load(open(path.join(fpath, "proteins.txt")), object_pairs_hook=OrderedDict)
        else:
            proteins = json.load(open(path.join(fpath, "proteins_hhmake.txt")), object_pairs_hook=OrderedDict)

        if sys.version_info > (3, 0):
            Y = pickle.load(open(path.join(fpath, "Y"), "rb"), encoding='latin1')
        else:
            Y = pickle.load(open(path.join(fpath, "Y"), "rb"))

        XT = self.process_proteins(proteins)

        XD = self.process_ligands(ligands)

        Y = np.mat(np.copy(Y))
        XD, XT, Y = np.asarray(XD), np.asarray(XT), np.asarray(Y)
        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

        return prepare_interaction_pairs(XD, XT, Y, label_row_inds, label_col_inds)

    def parse_csv(self, with_label):
        if with_label:
            dtc_train = pd.DataFrame()
            for file in glob.glob(path.join(self.fpath,'train*')):
                dtc_train = pd.concat((dtc_train, pd.read_csv(file)), axis=0, ignore_index=True)
        else:
            dtc_train = pd.DataFrame()
            for file in glob.glob(path.join(self.fpath,'test*')):
                dtc_train = pd.concat((dtc_train, pd.read_csv(file)), axis=0, ignore_index=True)

        dtc_train.columns = ['smiles' if col == 'Compound_SMILES' else col for col in dtc_train.columns]

        for ind in dtc_train[dtc_train['smiles'].str.contains('\n')].index:
            dtc_train.loc[ind, 'smiles'] = dtc_train.loc[ind, 'smiles'].split('\n')[0]

        n_samples = dtc_train['smiles'].shape[0]

        XD, XT = [1 for i in range(n_samples)], [1 for i in range(n_samples)]

        XD_processed = self.process_ligands(np.asarray(dtc_train['smiles'].unique().tolist()))
        indices_by_smiles = {}
        for ind, smiles in enumerate(dtc_train['smiles'].tolist()):
            if smiles not in indices_by_smiles:
                indices_by_smiles[smiles] = []

            indices_by_smiles[smiles].append(ind)

        for i, smiles in enumerate(np.asarray(dtc_train['smiles'].unique().tolist())):
            indices = indices_by_smiles[smiles]

            for ind in indices:
                XD[ind] = XD_processed[i]

        if 'fasta' not in dtc_train.columns:
            dtc_train = get_kinase_domains(dtc_train)

        XT_processed = self.process_proteins(np.asarray(dtc_train['fasta'].unique().tolist()))
        indices_by_fasta = {}
        for ind, fasta in enumerate(dtc_train['fasta'].tolist()):
            if fasta not in indices_by_fasta:
                indices_by_fasta[fasta] = []

            indices_by_fasta[fasta].append(ind)

        for i, fasta_seq in enumerate(dtc_train['fasta'].unique().tolist()):
            indices = indices_by_fasta[fasta_seq]
            for ind in indices:
                XT[ind] = XT_processed[i]

        assert len(XD) == len(XT) and len(XT) == dtc_train['smiles'].shape[0]
        assert np.asarray(XT).shape[1] == self.SEQLEN
        assert np.asarray(XD).shape[1] == self.SMILEN

        if with_label:
            return XD, XT, dtc_train['standard_value'].values
        else:
            return XD, XT

    def process_ligands(self, ligands):
        XD = []

        if self.drug_format == 'labeled_smiles':
            if type(ligands) == OrderedDict:
                iterator = ligands.keys()
            else:
                iterator = range(ligands.shape[0])

            for d in iterator:
                XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        elif self.drug_format == 'mol2vec':
            from rdkit.Chem import PandasTools
            from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
            from gensim.models import word2vec

            word2vec_model = word2vec.Word2Vec.load(self.mol2vec_model_path)
            df_ligands = pd.DataFrame({'smiles': ligands})

            PandasTools.AddMoleculeColumnToFrame(df_ligands, 'smiles', 'ROMol')
            dtc_train = df_ligands[df_ligands['ROMol'].notnull()]
            dtc_train.loc[:, 'mol-sentence'] = dtc_train.apply(
                lambda x: MolSentence(mol2alt_sentence(x['ROMol'], self.mol2vec_radius)), axis=1)
            XD = sentences2vec(dtc_train['mol-sentence'], word2vec_model, unseen='UNK')

        return XD

    def process_proteins(self, proteins):
        XT = []
        if type(proteins) == OrderedDict:
            iterator = proteins.keys()
        else:
            iterator = range(proteins.shape[0])

        if self.protein_format == 'sequence':
            for t in iterator:
                XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
        elif self.protein_format == 'pssm':
            for t in iterator:
                XT.append(get_PSSM(self.dataset_path, proteins[t], self.SEQLEN))
        elif self.protein_format == 'biovec':
            import biovec
            pv = biovec.models.load_protvec(self.biovec_model_path)
            for t in iterator:
                cur_vec = pv.to_vecs(proteins[t])
                cur_vec = np.concatenate((cur_vec[0], cur_vec[1], cur_vec[2]), axis=0)

                XT.append(cur_vec)
        else:
            raise NotImplementedError()

        return XT

def get_PSSM(dataset_path, data_file, max_seq_len):
    pssm = np.loadtxt(path.join(dataset_path, 'davis_dtc', 'hhmake_pssm', data_file.split('/')[-1]))[:max_seq_len, :]

    half_left = int((max_seq_len-pssm.shape[0])/2)
    half_right = max_seq_len-pssm.shape[0] - half_left

    pssm = np.pad(pssm, [(half_left, half_right), (0, 0) ], 'constant')

    return pssm

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(np.array(drug))

        target=XT[cols[pair_ind]]
        targets.append(np.array(target))

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    #drug_data = np.asarray(np.stack(drugs))
    #target_data = np.asarray(np.stack(targets))

    #assert drug_data.shape[0] == target_data.shape[0] and target_data.shape[0] == len(affinity)

    return drugs, targets, affinity

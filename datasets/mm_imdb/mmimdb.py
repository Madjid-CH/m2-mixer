import os
import sys
from typing import List

import h5py
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from omegaconf import DictConfig
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.transforms import RuinModality
from utils.projection import Projection


def _get_data_len(stage):
    if stage == 'train':
        return 80
    elif stage == 'test':
        return 20
    elif stage == 'dev':
        return 10


def _split_offset(stage):
    if stage == 'train':
        return 0
    elif stage == 'dev':
        return 80
    else:
        return 90


class MMIMDBDataset(Dataset):

    def __init__(self, root_dir, tokenizer, projection, max_seq_len, feat_dim=100, stage='train', transform=None):
        """
        Args:
            root_dir (string): Path to where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        super().__init__()
        self.root_dir = root_dir
        self.images, self.features, self.labels = self._setup_data()

        self.len_data = _get_data_len(stage)

        self.transform = transform

        self.stage = stage
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len

        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx: int) -> dict:
        idx += _split_offset(self.stage)
        image = Image.fromarray(self._get_image_from_dataset(idx)).convert('RGB')
        label = self.labels[idx]
        text = self.generate_text_from_sequence(self.features[idx])

        text_length = text.count(' ') + 1

        sample = {'image': image, 'text': text, 'label': label, 'textlen': text_length}

        if self.transform:
            for m in self.transform:
                if m == 'image':
                    sample[m] = self.transform[m](sample[m])
                elif m == 'multimodal':
                    sample = self.transform[m](sample)
                else:
                    sample[m] = self.transform[m](sample[m])

        fields = sample['text'].split('\t')
        words = self.get_words(fields)
        features = self.project_features(words)
        sample['text'] = features

        return sample

    def _get_image_from_dataset(self, idx):
        return self.images[idx].transpose(1, 2, 0).astype(np.uint8)

    def project_features(self, words: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens):
            tokens[index].append(token)
        features = self.projection(tokens)
        padded_features = np.pad(features, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_features

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields[0]))][
               :self.max_seq_len]

    def generate_text_from_sequence(self, sequence):
        data = np.load(f"{self.root_dir}/metadata.npy", allow_pickle=True).item()
        lookup = data['ix_to_word']
        return ' '.join([lookup[word] for word in sequence])

    def _setup_data(self):
        h5_file = h5py.File(f"{self.root_dir}/sample_file.h5", 'r')
        images = h5_file['images'][:]
        labels = h5_file['genres'][:]
        texts = h5_file['sequences'][:]
        return images, texts, labels


class MMIMDBDatasetWithFeatures(MMIMDBDataset):
    def __init__(self, root_dir, _tokenizer, _projection, _max_seq_len, _feat_dim=100, stage='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.images, self.features, self.labels = self._setup_data()
        self.len_data = _get_data_len(stage)
        self.transform = transform
        self.stage = stage

    def _setup_data(self):
        h5_file = h5py.File(f"{self.root_dir}/sample_file.h5", 'r')
        begin = _split_offset(self.stage)
        end = begin + _get_data_len(self.stage)
        images = h5_file['images'][begin:end]
        labels = h5_file['genres'][begin:end]
        features = h5_file['features'][begin:end]
        return images, features, labels

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx: int) -> dict:
        image = Image.fromarray(self._get_image_from_dataset(idx)).convert('RGB')
        label = self.labels[idx]
        features = self.features[idx]
        sample = {'image': image, 'text': features, 'label': label}
        if self.transform:
            sample = self._apply_transformation(sample)
        return sample

    def _apply_transformation(self, sample):
        for m in self.transform:
            if m == 'image':
                sample[m] = self.transform[m](sample[m])
            elif m == 'multimodal':
                sample = self.transform[m](sample)
            else:
                sample[m] = self.transform[m](sample[m])
        return sample

    def _get_image_from_dataset(self, idx):
        return self.images[idx].transpose(1, 2, 0).astype(np.uint8)


class MMIMDBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, vocab: DictConfig, projection: DictConfig,
                 max_seq_len: int, mmimdb_dataset="mmimdbdataset", **kwargs):
        super().__init__(**kwargs)
        self.padded_features = None
        self.train_set = None
        self.eval_set = None
        self.test_set = None
        self.mmimdb_dataset = getattr(sys.modules[__name__], mmimdb_dataset)
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_cfg = vocab
        self.projection = Projection(vocab.vocab_path, projection.feature_size, projection.window_size)
        self.tokenizer = BertWordPieceTokenizer(**vocab.tokenizer)

    def setup(self, stage: str = None):
        train_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]),
            multimodal=T.RandomApply([RuinModality(p=0.3)], p=0.6))

        val_test_transforms = dict(image=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

        self.train_set = self.mmimdb_dataset(os.path.join(self.data_dir), stage='train', tokenizer=self.tokenizer,
                                             projection=self.projection, max_seq_len=self.max_seq_len,
                                             transform=train_transforms)
        self.eval_set = self.mmimdb_dataset(os.path.join(self.data_dir), stage='dev', tokenizer=self.tokenizer,
                                            projection=self.projection, max_seq_len=self.max_seq_len,
                                            transform=val_test_transforms)
        self.test_set = self.mmimdb_dataset(os.path.join(self.data_dir), stage='test', tokenizer=self.tokenizer,
                                            projection=self.projection, max_seq_len=self.max_seq_len,
                                            transform=val_test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)


def generate_word_embeddings_with_bert(text):
    from transformers import BertTokenizer, BertModel
    import torch
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = model_output.last_hidden_state
    return embeddings

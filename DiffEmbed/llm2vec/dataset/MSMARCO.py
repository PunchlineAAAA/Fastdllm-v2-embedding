import hashlib
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import tiktoken
from accelerate.logging import get_logger
from datasets import load_dataset

from ..instruction import has_instruction_conflict, label_query_instruction
from .dataset import DataSample, Dataset, TrainSample

logger = get_logger(__name__, log_level="INFO")


CACHE_VERSION = "v1"
CACHE_ROOT = Path("cache/processed/msmarco")
LOCK_POLL_SECONDS = 2.0


def gpt2_token_count(text):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return len(tokens)


class MSMARCO(Dataset):
    def __init__(
        self,
        dataset_name: str = "msmarco-w-instructions",
        split: str = "train",
        file_path: str = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        aug_file_path: str = None,
        domain: str = "all",
        task: str = "all",
        add_e5: bool = False,
        enable_instruction_labels: bool = False,
        enable_instruction_conflicting_negatives: bool = False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.train_n_passages = 16
        self.neg_num = self.train_n_passages - 1
        self.enable_instruction_labels = enable_instruction_labels
        self.enable_instruction_conflicting_negatives = enable_instruction_conflicting_negatives

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def _cache_key(self, file_path: str) -> str:
        payload = {
            "version": CACHE_VERSION,
            "dataset_name": self.dataset_name,
            "file_path": file_path,
            "split": self.split,
            "separator": self.separator,
            "neg_num": self.neg_num,
            "enable_instruction_labels": self.enable_instruction_labels,
            "enable_instruction_conflicting_negatives": self.enable_instruction_conflicting_negatives,
            "sample_limit": 16_000,
        }
        return hashlib.md5(repr(sorted(payload.items())).encode("utf-8")).hexdigest()

    def _cache_paths(self, file_path: str):
        cache_key = self._cache_key(file_path)
        cache_dir = CACHE_ROOT
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.pkl"
        lock_path = cache_dir / f"{cache_key}.lock"
        return cache_path, lock_path

    def _load_cached_payload(self, cache_path: Path):
        with cache_path.open("rb") as fin:
            return pickle.load(fin)

    def _save_cached_payload(self, cache_path: Path, payload):
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as fout:
            pickle.dump(payload, fout, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)

    def _materialize_samples(self, file_path: str):
        logger.info(f"Loading data from {file_path}...")
        dataset = load_dataset(file_path)[self.split]
        dataset = dataset.select(range(16_000))

        instruction = "Retrieve text based on user query"
        all_samples = []

        for id_, example in enumerate(dataset):
            raw_query = example["query"]
            query = f"{instruction}; {self.separator}{raw_query}"
            query_label = label_query_instruction(raw_query, separator=self.separator)

            pos = None
            for pos_passage in example["positive_passages"]:
                pos = self.separator + self._join_passage(pos_passage)
                break

            new_negative_passages = [self.separator + self._join_passage(neg) for neg in example["new_negatives"]]
            negative_passages = [self.separator + self._join_passage(neg) for neg in example["negative_passages"]]
            negatives = self._select_negatives(query_label, new_negative_passages, negative_passages)

            metadata = query_label.to_dict() if self.enable_instruction_labels else {}
            all_samples.append(
                DataSample(
                    id_=id_,
                    query=query,
                    positive=pos,
                    task_name="msmarco-w-instruction",
                    batch_negatives=negatives,
                    metadata=metadata,
                )
            )

        query_avg_len, doc_avg_len, num_neg = [], [], []
        for sample in all_samples:
            query_avg_len.append(gpt2_token_count(sample.query))
            doc_avg_len.append(gpt2_token_count(sample.positive))
            num_neg.append(len(sample.batch_negatives))
            for negative in sample.batch_negatives:
                doc_avg_len.append(gpt2_token_count(negative))

        return {
            "data": all_samples,
            "stats": {
                "query_len_mean": float(np.mean(query_avg_len)),
                "query_len_std": float(np.std(query_avg_len, ddof=1)),
                "doc_len_mean": float(np.mean(doc_avg_len)),
                "doc_len_std": float(np.std(doc_avg_len, ddof=1)),
                "avg_number_of_negatives": float(np.mean(num_neg)),
            },
        }

    def _acquire_lock(self, lock_path: Path):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _print_stats(self, stats):
        print("query len mean: ", stats["query_len_mean"])
        print("query len std: ", stats["query_len_std"])
        print("doc len mean: ", stats["doc_len_mean"])
        print("doc len std: ", stats["doc_len_std"])
        print("avg number of negatives: ", stats["avg_number_of_negatives"])

    def _join_passage(self, passage):
        return passage["title"] + " " + passage["text"] if "title" in passage else passage["text"]

    def _select_negatives(self, query_label, new_negative_passages, negative_passages):
        negatives_first_n = min(3, len(new_negative_passages))
        seeded_negatives = new_negative_passages[:negatives_first_n]
        add_neg_num = self.neg_num - len(seeded_negatives)
        if add_neg_num <= 0:
            return seeded_negatives[: self.neg_num]

        if not self.enable_instruction_conflicting_negatives:
            pool = negative_passages if negative_passages else new_negative_passages
            sampler = random.choices if len(pool) < add_neg_num else random.sample
            return seeded_negatives + sampler(pool, k=add_neg_num)

        conflicting, fallback = [], []
        for negative in negative_passages:
            negative_label = label_query_instruction(
                negative, separator=self.separator
            )
            if has_instruction_conflict(query_label, negative_label):
                conflicting.append(negative)
            else:
                fallback.append(negative)

        selected = list(seeded_negatives)
        if conflicting:
            take = min(add_neg_num, len(conflicting))
            selected.extend(random.sample(conflicting, k=take) if len(conflicting) > take else conflicting)
            add_neg_num -= take
        if add_neg_num > 0:
            pool = fallback if fallback else negative_passages if negative_passages else new_negative_passages
            sampler = random.choices if len(pool) < add_neg_num else random.sample
            selected.extend(sampler(pool, k=add_neg_num))
        return selected

    def load_data(self, file_path: str = None):
        assert self.split == "train"
        cache_path, lock_path = self._cache_paths(file_path)

        payload = None
        if cache_path.exists():
            logger.info(f"Loading preprocessed MSMARCO cache from {cache_path}...")
            payload = self._load_cached_payload(cache_path)
        elif self._acquire_lock(lock_path):
            try:
                payload = self._materialize_samples(file_path)
                self._save_cached_payload(cache_path, payload)
            finally:
                if lock_path.exists():
                    lock_path.unlink()
        else:
            logger.info(f"Waiting for preprocessed MSMARCO cache at {cache_path}...")
            while not cache_path.exists():
                if not lock_path.exists() and self._acquire_lock(lock_path):
                    try:
                        payload = self._materialize_samples(file_path)
                        self._save_cached_payload(cache_path, payload)
                    finally:
                        if lock_path.exists():
                            lock_path.unlink()
                    break
                time.sleep(LOCK_POLL_SECONDS)
            if payload is None:
                payload = self._load_cached_payload(cache_path)

        all_samples = payload["data"]
        if self.shuffle_individual_datasets:
            random.shuffle(all_samples)
            logger.info("Shuffling samples...")

        self.data = all_samples
        logger.info(f"Loaded {len(self.data)} samples.")
        self._print_stats(payload["stats"])

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive] + sample.batch_negatives,
                label=1.0,
                metadata=sample.metadata,
            )
        elif self.split == "validation":
            assert False, "msmarco-w-instructions does not have a validation split."

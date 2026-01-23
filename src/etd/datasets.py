from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datasets

from etd.config import DatasetConfig


@dataclass
class SplitDatasets:
    train: Any
    validation: Any
    test: Any


def load_splits(cfg: DatasetConfig) -> SplitDatasets:
    load_dataset = getattr(datasets, "load_dataset")
    dataset = load_dataset(cfg.name, cfg.config)

    if cfg.split not in dataset:
        raise ValueError(f"Missing split {cfg.split} in {cfg.name}")
    if cfg.validation_split not in dataset:
        raise ValueError(f"Missing split {cfg.validation_split} in {cfg.name}")
    if cfg.test_split not in dataset:
        raise ValueError(f"Missing split {cfg.test_split} in {cfg.name}")

    train = dataset[cfg.split]
    validation = dataset[cfg.validation_split]
    test = dataset[cfg.test_split]

    train = _select_limit(train, cfg.train_limit)
    validation = _select_limit(validation, cfg.validation_limit)
    test = _select_limit(test, cfg.test_limit)

    return SplitDatasets(train=train, validation=validation, test=test)


def prepare_text(dataset: Any, cfg: DatasetConfig) -> Any:
    if cfg.text_field not in dataset.column_names:
        raise ValueError(f"Missing text field {cfg.text_field} in dataset columns")
    if cfg.id_field and cfg.id_field not in dataset.column_names:
        raise ValueError(f"Missing id field {cfg.id_field} in dataset columns")

    def _map(example: dict) -> dict:
        output = {"text": example[cfg.text_field]}
        if cfg.id_field:
            output["doc_id"] = example[cfg.id_field]
        return output

    return dataset.map(_map, remove_columns=list(dataset.column_names))


def _select_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))

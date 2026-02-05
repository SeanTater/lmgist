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

    # Auto-split single-split datasets into train/validation/test deterministically.
    if (
        cfg.validation_split == cfg.split
        and cfg.test_split == cfg.split
        and (cfg.validation_ratio or cfg.test_ratio)
    ):
        train = dataset[cfg.split]
        validation_ratio = cfg.validation_ratio or 0.0
        test_ratio = cfg.test_ratio or 0.0
        holdout_ratio = validation_ratio + test_ratio
        if holdout_ratio <= 0.0:
            raise ValueError("validation_ratio/test_ratio must be > 0 when auto-splitting")
        if holdout_ratio >= 1.0:
            raise ValueError("validation_ratio + test_ratio must be < 1.0")
        # Hold out a combined validation+test slice, then split that into two parts.
        split = train.train_test_split(
            test_size=holdout_ratio,
            seed=cfg.shuffle_seed,
            shuffle=True,
        )
        train = split["train"]
        holdout = split["test"]
        if test_ratio > 0.0:
            test_fraction = test_ratio / holdout_ratio
            # Reuse the same seed to make the two-stage split reproducible.
            holdout_split = holdout.train_test_split(
                test_size=test_fraction,
                seed=cfg.shuffle_seed,
                shuffle=True,
            )
            validation = holdout_split["train"]
            test = holdout_split["test"]
        else:
            validation = holdout
            test = holdout.select([])
    else:
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

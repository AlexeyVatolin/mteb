from __future__ import annotations

import argparse
from typing import Any

import datasets

import mteb
from mteb.cli import add_task_selection_args


def flatten(xss):
    return [x for xs in xss for x in xs]


def get_text_columns(task: mteb.AbsTask) -> list[str]:
    if isinstance(task, mteb.AbsTaskBitextMining):
        return list({p for pair in task.get_pairs(task.parallel_subsets) for p in pair})
    elif isinstance(task, mteb.AbsTaskPairClassification) or isinstance(task, mteb.AbsTaskSTS):
        return ["sentence1", "sentence2"]
    elif isinstance(task, mteb.AbsTaskClassification) or isinstance(
        task, mteb.AbsTaskMultilabelClassification
    ):
        return ["text"]
    elif isinstance(task, mteb.AbsTaskClustering) or isinstance(task, mteb.AbsTaskClusteringFast):
        return ["sentences"]
    raise ValueError(f"Unsuppported task type {task}")


def is_trainable_task(task: mteb.AbsTask) -> bool:
    return isinstance(task, mteb.AbsTaskClassification) or isinstance(
        task, mteb.AbsTaskMultilabelClassification
    )


def get_splits(task: mteb.AbsTask) -> list[str]:
    if is_trainable_task(task):
        return ["train"] + task.metadata.eval_splits
    return task.metadata.eval_splits


def verify_non_empty_texts(
    texts: list[str], dataset_name: str, hf_subset: str, split: str, column_name: str
) -> bool:
    if num_empty := sum(len(doc.strip()) == 0 for doc in texts):
        print(
            f"{num_empty} empty documents found in task {dataset_name}, subset {hf_subset}, split {split}, column_name {column_name}"
        )
        return False
    return True


def verify_duplicates(
    texts: list[str], dataset_name: str, hf_subset: str, split: str, column_name: str
) -> bool:
    if num_duplicates := len(texts) - len({text.strip() for text in texts}):
        print(
            f"{num_duplicates} duplicated documents found in task {dataset_name}, subset {hf_subset}, split {split}, column_name {column_name}"
        )
        return False
    return True


def get_ds_unique_examples(task: mteb.AbsTask, ds_split) -> set[tuple[str | tuple]]:
    text_columns = get_text_columns(task)
    return {
        tuple(
            [row[col] for col in text_columns]
            + [tuple(row["label"]) if isinstance(row["label"], list) else row["label"]]
        )
        for row in ds_split
    }


def verify_leakage(task: mteb.AbsTask, hf_subset: str, ds_subset) -> bool:
    is_valid = True
    train_examples = get_ds_unique_examples(task, ds_subset["train"])
    for split in task.metadata.eval_splits:
        split_examples = get_ds_unique_examples(task, ds_subset[split])
        if num_duplicates := len(train_examples.intersection(split_examples)):
            print(
                f"{num_duplicates} leaked documents found in task {task.metadata.name}, subset {hf_subset}, split {split}"
            )
            is_valid = False
    return is_valid


def get_subset(ds, subset_name):
    return ds if subset_name == "default" else ds[subset_name]


def check_splits(ds, task: mteb.AbsTask, hf_subset, column_name):
    is_valid = True
    for split in get_splits(task):
        split_texts = ds[split]
        if isinstance(split_texts[0], list):
            split_texts = flatten(split_texts)
        is_valid &= verify_non_empty_texts(
            split_texts, task.metadata.name, hf_subset, split, column_name
        )
        is_valid &= verify_duplicates(
            split_texts, task.metadata.name, hf_subset, split, column_name
        )
    return is_valid


def validate_task(task: mteb.AbsTask) -> bool:
    hf_subsets = (
        ["default"]
        if not task.is_multilingual
        or isinstance(task, mteb.AbsTaskBitextMining)
        and task.parallel_subsets
        else list(task.dataset)
    )
    is_valid = True

    for hf_subset in hf_subsets:
        if isinstance(task, mteb.AbsTaskInstructionRetrieval):
            queries = {
                split: list(ds.values())
                for split, ds in get_subset(task.queries, hf_subset).items()
            }
            corpus = get_subset(task.corpus, hf_subset)
            corpus_texts = {
                split: [
                    row.get("title", "") + row.get("text", "") for row in corpus[split].values()
                ]
                for split in task.metadata.eval_splits
            }

            is_valid &= check_splits(queries, task, hf_subset, "queries")
            is_valid &= check_splits(corpus_texts, task, hf_subset, "corpus")

        elif isinstance(task, mteb.AbsTaskReranking):
            ds = get_subset(task.dataset, hf_subset)
            for column_name in ("query", "positive", "negative"):
                texts = {split: ds[split][column_name] for split in task.metadata.eval_splits}
                is_valid &= check_splits(texts, task, hf_subset, column_name)
        elif isinstance(task, mteb.AbsTaskRetrieval):
            queries = get_subset(task.queries, hf_subset)
            corpus = get_subset(task.corpus, hf_subset)
            query_texts = {
                split: [
                    query if isinstance(query, str) else "".join(queries)
                    for query in queries[split].values()
                ]
                for split in task.metadata.eval_splits
            }
            corpus_texts = {
                split: [
                    row.get("title", "") + row.get("text", "") for row in corpus[split].values()
                ]
                for split in task.metadata.eval_splits
            }

            is_valid &= check_splits(query_texts, task, hf_subset, "query")
            is_valid &= check_splits(corpus_texts, task, hf_subset, "corpus")

        else:
            ds = get_subset(task.dataset, hf_subset)
            if is_trainable_task(task):
                is_valid &= verify_leakage(task, hf_subset, ds)
            text_columns = get_text_columns(task)
            if text_columns:
                for column in text_columns:
                    splits = {
                        split: ds[split][0][column]
                        if isinstance(ds[split], list)
                        else ds[split][column]
                        for split in ds
                    }
                    is_valid &= check_splits(splits, task, hf_subset, column)
            else:
                raise ValueError(f"No validator for task {task.metadata.name}")

    return is_valid


def check_duplicates(dataset: datasets.DatasetDict) -> bool:
    for split, data in dataset.items():
        if len(data) != len(set(data["text"])):
            print(f"Duplicate documents found in {split} split.")
            return False
    return True


def check_leakage(train_data: list[str], test_data: list[str]) -> bool:
    train_set = set(train_data)
    test_set = set(test_data)
    leakage = train_set.intersection(test_set)
    if leakage:
        print(f"Leakage found between train and test sets: {leakage}")
        return False
    return True


def compute_metrics(dataset: datasets.DatasetDict) -> dict[str, Any]:
    metrics = {}
    for split, data in dataset.items():
        lengths = [len(doc.split()) for doc in data["text"]]
        metrics[split] = {
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "num_documents": len(data),
        }
    return metrics


def validate_dataset(args: argparse.Namespace) -> None:
    tasks: tuple[mteb.AbsTask] = mteb.get_tasks(
        categories=args.categories,
        task_types=args.task_types,
        languages=args.languages,
        tasks=args.tasks,
    )

    if not tasks:
        print("No tasks found with the given criteria.")
        return

    task: mteb.AbsTask
    for task in tasks:
        print(f"Validating {task.metadata.name}")
        if task.superseded_by:
            print(f"Task {task.metadata.name} is superseded by {task.superseded_by}, skipping..")
            continue
        task.load_data()

        task_valid = validate_task(task)

        if task_valid:
            print(f"Task {task.metadata.name} validated, no problems found")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for MTEB.")
    add_task_selection_args(parser)

    args = parser.parse_args()
    validate_dataset(args)


if __name__ == "__main__":
    main()

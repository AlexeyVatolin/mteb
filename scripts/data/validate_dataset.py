from __future__ import annotations

import argparse
from typing import Any

import datasets

import mteb
from mteb.cli import add_task_selection_args


def flatten(xss):
    return [x for xs in xss for x in xs]


def get_text_columns(task: mteb.AbsTask):
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


def get_splits(task: mteb.AbsTask) -> list[str]:
    if (
        isinstance(task, mteb.AbsTaskPairClassification)
        or isinstance(task, mteb.AbsTaskClassification)
        or isinstance(task, mteb.AbsTaskMultilabelClassification)
    ):
        return ["train"] + task.metadata.eval_splits
    return task.metadata.eval_splits


def has_empty_texts(texts: list[str], dataset_name: str, hf_subset: str, split: str) -> bool:
    if num_empty := sum(len(doc.strip()) == 0 for doc in texts):
        print(
            f"{num_empty} empty documents found in task {dataset_name}, subset {hf_subset}, split {split}"
        )
        return False
    return True


def has_duplicates(texts: list[str], dataset_name: str, hf_subset: str, split: str) -> bool:
    if num_duplicates := len(texts) - len({text.strip() for text in texts}):
        print(
            f"{num_duplicates} duplicated documents found in task {dataset_name}, subset {hf_subset}, split {split}"
        )
        return False
    return True


def get_subset(ds, subset_name):
    return ds if subset_name == "default" else ds[subset_name]


def check_splits(ds, task: mteb.AbsTask, hf_subset):
    success = True
    for split in get_splits(task):
        success &= has_empty_texts(ds[split], task.metadata.name, hf_subset, split)
    return success


def check_empty_documents(task: mteb.AbsTask) -> bool:
    hf_subsets = (
        ["default"]
        if not task.is_multilingual
        or isinstance(task, mteb.AbsTaskBitextMining)
        and task.parallel_subsets
        else list(task.dataset)
    )
    success = True

    for hf_subset in hf_subsets:
        if isinstance(task, mteb.AbsTaskInstructionRetrieval):
            queries = get_subset(task.queries, hf_subset)
            corpus = get_subset(task.corpus, hf_subset)
            corpus_texts = {
                split: [row.get("title", "") + row.get("text", "") for row in corpus[split]]
                for split in task.metadata.eval_splits
            }

            success &= check_splits(queries, task, hf_subset)
            success &= check_splits(corpus_texts, task, hf_subset)

        elif isinstance(task, mteb.AbsTaskReranking):
            ds = get_subset(task.dataset, hf_subset)
            for split in task.metadata.eval_splits:
                success &= has_empty_texts(
                    ds[split]["query"], task.metadata.name, hf_subset, split
                )
                success &= has_empty_texts(
                    flatten(ds[split]["positive"]), task.metadata.name, hf_subset, split
                )
                success &= has_empty_texts(
                    flatten(ds[split]["negative"]), task.metadata.name, hf_subset, split
                )

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

            success &= check_splits(query_texts, task, hf_subset)
            success &= check_splits(corpus_texts, task, hf_subset)

        else:
            ds = get_subset(task.dataset, hf_subset)
            text_columns = get_text_columns(task)
            if text_columns:
                for column in text_columns:
                    success &= check_splits(
                        {split: ds[split][column] for split in ds}, task, hf_subset
                    )
            else:
                raise ValueError(f"No validator for task {task.metadata.name}")

    return success


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
        task_valid = True
        print(f"Validating {task.metadata.name}")
        task.load_data()

        task_valid &= check_empty_documents(task)

        # if not check_duplicates(dataset):
        #     print(f"Validation failed for {task}: Duplicate documents found.")
        #     continue

        # if "train" in dataset and "test" in dataset:
        #     if not check_leakage(dataset["train"]["text"], dataset["test"]["text"]):
        #         print(f"Validation failed for {task}: Leakage between train and test sets.")
        #         continue

        # metrics = compute_metrics(dataset)
        # print(f"Validation passed for {task}. Metrics:")
        # for split, split_metrics in metrics.items():
        #     print(f"{split} split: {split_metrics}")
        if task_valid:
            print(f"Task {task.metadata.name} validated, no problems found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset for MTEB.")
    add_task_selection_args(parser)

    args = parser.parse_args()
    validate_dataset(args)

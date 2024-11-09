from __future__ import annotations

import argparse
import csv
import traceback
from typing import Any

import datasets
from tqdm import tqdm

import mteb
from mteb.cli import add_task_selection_args

ProblemsList = list[tuple[str, str, str, str, str, int]]


def flatten(nested_list: list[list[Any]]) -> list[Any]:
    return [element for sublist in nested_list for element in sublist]


def get_text_columns(task: mteb.AbsTask) -> list[str]:
    if isinstance(task, mteb.AbsTaskBitextMining):
        return list({p for pair in task.get_pairs(task.parallel_subsets) for p in pair})
    elif isinstance(task, mteb.AbsTaskPairClassification) or isinstance(
        task, mteb.AbsTaskSTS
    ):
        return ["sentence1", "sentence2"]
    elif isinstance(task, mteb.AbsTaskClassification) or isinstance(
        task, mteb.AbsTaskMultilabelClassification
    ):
        return ["text"]
    elif isinstance(task, mteb.AbsTaskClustering) or isinstance(
        task, mteb.AbsTaskClusteringFast
    ):
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


def get_subset(
    ds: datasets.DatasetDict, subset_name: str
) -> datasets.DatasetDict | datasets.Dataset:
    return ds if subset_name == "default" else ds[subset_name]


def verify_non_empty_texts(
    texts: list[str],
    dataset_name: str,
    hf_subset: str,
    split: str,
    column_name: str,
    problems: ProblemsList,
) -> bool:
    if num_empty := sum(len(doc.strip()) == 0 for doc in texts):
        print(
            f"{num_empty} empty documents found in task {dataset_name}, subset {hf_subset}, split {split}, column_name {column_name}"
        )
        problems.append(
            ("empty_documents", dataset_name, hf_subset, split, column_name, num_empty)
        )
        return False
    return True


def verify_duplicates(
    texts: list[str],
    dataset_name: str,
    hf_subset: str,
    split: str,
    column_name: str,
    problems: ProblemsList,
) -> bool:
    if num_duplicates := len(texts) - len({text.strip() for text in texts}):
        print(
            f"{num_duplicates} duplicated documents found in task {dataset_name}, subset {hf_subset}, split {split}, column_name {column_name}"
        )
        problems.append(
            (
                "duplicated_documents",
                dataset_name,
                hf_subset,
                split,
                column_name,
                num_duplicates,
            )
        )
        return False
    return True


def verify_full_duplicate(
    task: mteb.AbsTask,
    hf_subset: str,
    ds_subset: datasets.DatasetDict,
    problems: ProblemsList,
) -> bool:
    text_columns = get_text_columns(task)
    if len(text_columns) < 2:
        return True

    is_valid = True
    for split in ds_subset:
        split_examples = get_ds_unique_examples(
            task, ds_subset[split], with_label=False
        )
        if num_duplicates := len(ds_subset[split]) - len(split_examples):
            print(
                f"{num_duplicates} fully duplicated documents found in task {task.metadata.name}, subset {hf_subset}, split {split}"
            )
            problems.append(
                (
                    "fully_duplicated_documents",
                    task.metadata.name,
                    hf_subset,
                    split,
                    "N/A",
                    num_duplicates,
                )
            )
            is_valid = False
    return is_valid


def get_ds_unique_examples(
    task: mteb.AbsTask, ds_split: datasets.Dataset, with_label: bool = True
) -> set[tuple[str | tuple]]:
    text_columns = get_text_columns(task)
    return {
        tuple(
            [row[col] for col in text_columns]
            + [tuple(row["label"]) if isinstance(row["label"], list) else row["label"]]
            if with_label
            else [row[col] for col in text_columns]
        )
        for row in ds_split
    }


def verify_leakage(
    task: mteb.AbsTask,
    hf_subset: str,
    ds_subset: datasets.DatasetDict,
    problems: ProblemsList,
) -> bool:
    is_valid = True
    train_examples = get_ds_unique_examples(task, ds_subset["train"])
    for split in task.metadata.eval_splits:
        split_examples = get_ds_unique_examples(task, ds_subset[split])
        if num_duplicates := len(train_examples.intersection(split_examples)):
            print(
                f"{num_duplicates} leaked documents found in task {task.metadata.name}, subset {hf_subset}, split {split}"
            )
            problems.append(
                (
                    "leaked_documents",
                    task.metadata.name,
                    hf_subset,
                    split,
                    "N/A",
                    num_duplicates,
                )
            )
            is_valid = False
    return is_valid


def save_problems_table(problems: ProblemsList, output_file: str):
    if problems:
        headers = [
            "Problem Name",
            "Dataset Name",
            "HF Subset",
            "Split",
            "Column Name",
            "Num Documents",
        ]
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(problems)


def check_splits(
    ds: dict[str, list[str]],
    task: mteb.AbsTask,
    hf_subset: str,
    column_name: str,
    problems: ProblemsList,
) -> bool:
    is_valid = True
    for split in get_splits(task):
        split_texts = ds[split]
        if isinstance(split_texts[0], list):
            split_texts = flatten(split_texts)
        is_valid &= verify_non_empty_texts(
            split_texts, task.metadata.name, hf_subset, split, column_name, problems
        )
        is_valid &= verify_duplicates(
            split_texts, task.metadata.name, hf_subset, split, column_name, problems
        )
    return is_valid


def validate_task(task: mteb.AbsTask, problems: ProblemsList) -> bool:
    hf_subsets = (
        ["default"]
        if not task.is_multilingual
        or isinstance(task, mteb.AbsTaskBitextMining)
        and task.parallel_subsets
        else list(task.hf_subsets)
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
                    row.get("title", "") + row.get("text", "")
                    for row in corpus[split].values()
                ]
                for split in task.metadata.eval_splits
                if corpus[split] is not None
            }

            is_valid &= check_splits(queries, task, hf_subset, "queries", problems)
            is_valid &= check_splits(corpus_texts, task, hf_subset, "corpus", problems)

        elif isinstance(task, mteb.AbsTaskReranking):
            ds = get_subset(task.dataset, hf_subset)
            for column_name in ("query", "positive", "negative"):
                texts = {
                    split: ds[split][column_name] for split in task.metadata.eval_splits
                }
                is_valid &= check_splits(texts, task, hf_subset, column_name, problems)
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
                    row.get("title", "") + row.get("text", "")
                    if isinstance(row, dict)
                    else row
                    for row in corpus[split].values()
                ]
                for split in task.metadata.eval_splits
                if corpus[split] is not None
            }

            is_valid &= check_splits(query_texts, task, hf_subset, "query", problems)
            is_valid &= check_splits(corpus_texts, task, hf_subset, "corpus", problems)

        else:
            ds = get_subset(task.dataset, hf_subset)
            if is_trainable_task(task):
                is_valid &= verify_leakage(task, hf_subset, ds, problems)
            is_valid &= verify_full_duplicate(task, hf_subset, ds, problems)
            text_columns = get_text_columns(task)
            if text_columns:
                for column in text_columns:
                    splits = {
                        split: ds[split][0][column]
                        if isinstance(ds[split], list)
                        else ds[split][column]
                        for split in ds
                    }
                    is_valid &= check_splits(splits, task, hf_subset, column, problems)
            else:
                raise ValueError(f"No validator for task {task.metadata.name}")

    return is_valid


def validate_dataset(args: argparse.Namespace) -> None:
    tasks: tuple[mteb.AbsTask] = mteb.get_tasks(
        categories=args.categories,
        task_types=args.task_types,
        languages=args.languages,
        tasks=args.tasks,
    )
    tasks = sorted(tasks, key=lambda task: task.metadata.name)

    if not tasks:
        print("No tasks found with the given criteria.")
        return

    problems: ProblemsList = []
    task: mteb.AbsTask
    progress = tqdm(tasks)
    for task in progress:
        progress.set_description(task.metadata.name)
        print(f"Validating {task.metadata.name}")
        if task.superseded_by:
            print(
                f"Task {task.metadata.name} is superseded by {task.superseded_by}, skipping.."
            )
            continue
        # if task.metadata.name in {"XStance"}:
        #     continue

        # if task.metadata.name < "XMarket":
        #     continue

        try:
            task.load_data()
            task_valid = validate_task(task, problems)
            if task_valid:
                print(f"Task {task.metadata.name} validated, no problems found")
        except Exception as e:
            print(task.metadata.name)
            print(e)
            print(traceback.format_exc())

        # if i > 10:
        #     break

    save_problems_table(problems, args.output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset for MTEB.")
    add_task_selection_args(parser)
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output CSV file where problems will be saved.",
    )

    args = parser.parse_args()
    validate_dataset(args)


if __name__ == "__main__":
    main()

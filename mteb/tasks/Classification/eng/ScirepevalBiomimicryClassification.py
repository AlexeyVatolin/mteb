import datasets

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ScirepevalBiomimicryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ScirepevalBiomimicryClassification",
        description="DBpedia14 is a dataset of English texts from Wikipedia articles, categorized into 14 non-overlapping classes based on their DBpedia ontology.",
        reference="https://arxiv.org/abs/1509.01626",
        dataset={
            "path": "allenai/scirepeval",
            "revision": "781d35d1bf87253b3dcd0fadcb82bfbee9c244f1",
            "name": "biomimicry",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-3.0",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @inproceedings{NIPS2015_250cf8b5,
            author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
            booktitle = {Advances in Neural Information Processing Systems},
            editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
            pages = {},
            publisher = {Curran Associates, Inc.},
            title = {Character-level Convolutional Networks for Text Classification},
            url = {https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
            volume = {28},
            year = {2015}
            }
        """,
        n_samples={"test": 70000},
        avg_character_length={"test": 281.40},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["samples_per_label"] = 100_000
        return metadata_dict

    @staticmethod
    def _format_text(row: dict) -> str:
        text = []
        for field in ["title", "abstract"]:
            if row.get(field):
                text.append(str(row[field]))
        return ("  ".join(text)).strip()

    def dataset_transform(self):
        test_dataset = datasets.load_dataset(
            "allenai/scirepeval_test",
            name="biomimicry",
            revision="7474d71febabf411f9006cb7b6cc895d57fdf48b",
        )
        splits = ["train", "test"]
        label_by_id = {
            split: {row["paper_id"]: row["label"] for row in test_dataset[split]}
            for split in splits
        }

        dataset_dict = {}
        for split in splits:
            dataset_dict[split] = (
                self.dataset["evaluation"]
                .filter(lambda row: row["doc_id"] in label_by_id[split])
                .map(
                    lambda row: {
                        "text": self._format_text(row),
                        "label": label_by_id[split][row["doc_id"]],
                    }
                )
                .select_columns(["text", "label"])
            )
        self.dataset = datasets.DatasetDict(dataset_dict)

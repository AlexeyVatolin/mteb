import random

import datasets
from tqdm import tqdm

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "ja": ["jpn-Jpan"],
}

NUM_SAMPLES = 2048


class ESCIRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="ESCIRetrieval",
        description="Amazon esci is a dataset consisting of retrieval queries and products information on Amazon. For each data, the relevance between query and product is annotated with E(Exact), S(Substitute), C(Complement), and I(Irrelevant).　Each relevance label is given a different score, allowing for more detailed scoring. We employed product titles and descriptions as product information and excluded data without descriptions.",
        reference="https://github.com/amazon-science/esci-data/",
        dataset={
            "path": "tasksource/esci",
            "revision": "8113b17a5d4099e20243282c926f1bc1a08a4d13",
        },
        type="Reranking",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2020-01-01", "2023-04-14"),  # supposition
        form=["written"],
        domains=["Web"],
        task_subtypes=None,
        license="Apache 2.0",
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@article{reddy2022shopping,
title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
author={Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
year={2022},
eprint={2206.06588},
archivePrefix={arXiv}
}
""",
        n_samples={lang: NUM_SAMPLES for lang in _LANGUAGES},
        avg_character_length=None,
    )

    @staticmethod
    def _sample_dataset(
        corpus: dict,
        queries: dict,
        relevant_docs: dict,
        seed: int,
        splits: list[str],
        n_samples: int = 2048,
    ):
        random.seed(seed)
        for lang in queries:
            for split in splits:
                if len(queries[lang][split]) <= n_samples:
                    continue
                query_ids = random.sample(list(queries[lang][split]), k=n_samples)
                queries[lang][split] = {
                    query_id: queries[lang][split][query_id] for query_id in query_ids
                }

                corpus_keys = {
                    product_id
                    for key in query_ids
                    for product_id in relevant_docs[lang][split][key]
                    if product_id in corpus[lang][split]
                }

                relevant_docs[lang][split] = {
                    query_id: relevant_docs[lang][split][query_id] for query_id in query_ids
                }
                corpus[lang][split] = {key: corpus[lang][split][key] for key in corpus_keys}
        return corpus, queries, relevant_docs

    @staticmethod
    def _filter_non_existing_relevant_docs(corpus: dict, relevant_docs: dict) -> dict:
        for lang in relevant_docs:
            for split in relevant_docs[lang]:
                remove_query_id = []
                for query_id in relevant_docs[lang][split]:
                    query_relevant = {
                        product_id: esci_label
                        for product_id, esci_label in relevant_docs[lang][split][query_id].items()
                        if product_id in corpus[lang][split]
                    }
                    if len(query_relevant) == 0:
                        remove_query_id.append(query_id)
                    else:
                        relevant_docs[lang][split][query_id] = query_relevant
                for query_id in remove_query_id:
                    del relevant_docs[lang][split][query_id]
        return relevant_docs

    @staticmethod
    def _filter_non_existing_corpus_docs(corpus: dict, relevant_docs: dict) -> dict:
        for lang in relevant_docs:
            for split in relevant_docs[lang]:
                product_ids = {
                    product_id
                    for query_id in relevant_docs[lang][split]
                    for product_id in relevant_docs[lang][split][query_id]
                }
                non_existent_docs = set(corpus[lang][split]).difference(product_ids)
                for product_id in non_existent_docs:
                    del corpus[lang][split][product_id]
        return corpus

    @staticmethod
    def _filter_non_existing_queries(queries: dict, relevant_docs: dict) -> dict:
        for lang in queries:
            for split in queries[lang]:
                non_existent_queries = set(queries).difference(set(relevant_docs[lang][split]))
                for query_id in non_existent_queries:
                    del queries[lang][split][query_id]
        return queries

    def load_data(self, **kwargs):
        product_locale_map = {"jp": "ja", "us": "en"}
        label_map = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        corpus = {lang: {self._EVAL_SPLIT: {}} for lang in _LANGUAGES}
        queries = {lang: {self._EVAL_SPLIT: {}} for lang in _LANGUAGES}
        relevant_docs = {lang: {self._EVAL_SPLIT: {}} for lang in _LANGUAGES}

        for example in tqdm(data, desc="Preparing data"):
            product_locale = example.get("product_locale")
            lang = product_locale_map.get(product_locale, product_locale)

            if example.get("query_id") and example.get("query"):
                query_id = str(example["query_id"])
                query_text = example["query"]

                if query_id not in queries[lang][self._EVAL_SPLIT]:
                    queries[lang][self._EVAL_SPLIT][query_id] = query_text

                product_id = example.get("product_id")
                esci_label = example.get("esci_label")
                if product_id and esci_label:
                    relevant_docs[lang][self._EVAL_SPLIT].setdefault(query_id, {})[product_id] = (
                        label_map[esci_label]
                    )

            if (
                example.get("product_id")
                and example.get("product_title")
                and example.get("product_description")
            ):
                product_id = example["product_id"]
                product_title = example["product_title"]
                product_description = example["product_description"]
                if product_id not in corpus[lang][self._EVAL_SPLIT]:
                    corpus[lang][self._EVAL_SPLIT][product_id] = {
                        "text": product_title + ": " + product_description,
                    }

        self.corpus, self.queries, self.relevant_docs = corpus, queries, relevant_docs
        self.corpus, self.queries, self.relevant_docs = self._sample_dataset(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            seed=self.seed,
            splits=[self._EVAL_SPLIT],
        )
        print(
            "Total relevant docs",
            sum(
                len(self.relevant_docs[lang][split][query_id])
                for lang in self.relevant_docs
                for split in self.relevant_docs[lang]
                for query_id in self.relevant_docs[lang][split]
            ),
        )
        self.relevant_docs = self._filter_non_existing_relevant_docs(
            self.corpus, self.relevant_docs
        )
        print(
            "Total relevant docs",
            sum(
                len(self.relevant_docs[lang][split][query_id])
                for lang in self.relevant_docs
                for split in self.relevant_docs[lang]
                for query_id in self.relevant_docs[lang][split]
            ),
        )
        print(
            "Total corpus docs",
            sum(
                len(self.corpus[lang][split])
                for lang in self.corpus
                for split in self.corpus[lang]
            ),
        )
        self.corpus = self._filter_non_existing_corpus_docs(self.corpus, self.relevant_docs)
        print(
            "Total corpus docs",
            sum(
                len(self.corpus[lang][split])
                for lang in self.corpus
                for split in self.corpus[lang]
            ),
        )
        self.queries = self._filter_non_existing_queries(self.queries, self.relevant_docs)

        self.data_loaded = True

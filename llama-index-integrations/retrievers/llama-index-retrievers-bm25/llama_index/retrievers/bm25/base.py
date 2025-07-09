import json
import logging
import os
import numpy as np

from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)

import bm25s
import Stemmer


logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {
    "similarity_top_k": "similarity_top_k",
    "_verbose": "verbose",
    "filters": "filters",
}

DEFAULT_PERSIST_FILENAME = "retriever.json"


def _build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[int], Dict[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[int], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.filters if metadata_filters else []
    if not filter_list or not metadata_filters:
        return lambda _: True

    filter_condition = metadata_filters.condition

    def _process_filter_match(
        operator: FilterOperator, value: Any, metadata_value: Any
    ) -> bool:
        if metadata_value is None:
            return False
        if operator == FilterOperator.EQ:
            return metadata_value == value
        if operator == FilterOperator.NE:
            return metadata_value != value
        if operator == FilterOperator.GT:
            return metadata_value > value
        if operator == FilterOperator.GTE:
            return metadata_value >= value
        if operator == FilterOperator.LT:
            return metadata_value < value
        if operator == FilterOperator.LTE:
            return metadata_value <= value
        if operator == FilterOperator.IN:
            return metadata_value in value
        if operator == FilterOperator.NIN:
            return metadata_value not in value
        if operator == FilterOperator.CONTAINS:
            return value in metadata_value
        if operator == FilterOperator.TEXT_MATCH:
            return value.lower() in metadata_value.lower()
        if operator == FilterOperator.ALL:
            return all(val in metadata_value for val in value)
        if operator == FilterOperator.ANY:
            return any(val in metadata_value for val in value)
        raise ValueError(f"Invalid operator: {operator}")

    def filter_fn(idx: int) -> bool:
        metadata = metadata_lookup_fn(idx)

        filter_matches_list = []
        for filter_ in filter_list:
            if isinstance(filter_, MetadataFilters):
                raise ValueError("Nested MetadataFilters are not supported.")

            metadata_value = metadata.get(filter_.key, None)
            if filter_.operator == FilterOperator.IS_EMPTY:
                filter_matches = (
                    metadata_value is None
                    or metadata_value == ""
                    or metadata_value == []
                )
            else:
                filter_matches = _process_filter_match(
                    operator=filter_.operator,
                    value=filter_.value,
                    metadata_value=metadata_value,
                )

            filter_matches_list.append(filter_matches)

        if filter_condition == FilterCondition.AND:
            return all(filter_matches_list)
        elif filter_condition == FilterCondition.OR:
            return any(filter_matches_list)
        else:
            raise ValueError(f"Invalid filter condition: {filter_condition}")

    return filter_fn


class BM25Retriever(BaseRetriever):
    r"""
    A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        stemmer (Stemmer.Stemmer, optional):
            The stemmer to use. Defaults to an english stemmer.
        language (str, optional):
            The language to use for stopword removal. Defaults to "en".
        existing_bm25 (bm25s.BM25, optional):
            An existing BM25 object to use. If not provided, nodes must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        filters (MetadataFilters, optional):
            Metadata filters to apply at query time. Defaults to None.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        token_pattern (str, optional):
            The token pattern to use. Defaults to (?u)\\b\\w\\w+\\b.
        skip_stemming (bool, optional):
            Whether to skip stemming. Defaults to False.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.

    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25s.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        filters: Optional[MetadataFilters] = None,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
    ) -> None:
        self.stemmer = stemmer or Stemmer.Stemmer("english")
        self.similarity_top_k = similarity_top_k
        self.token_pattern = token_pattern
        self.skip_stemming = skip_stemming
        self._filters = filters

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            self.corpus = [node_to_metadata_dict(node) for node in nodes]

            corpus_tokens = bm25s.tokenize(
                [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
                stopwords=language,
                stemmer=self.stemmer if not skip_stemming else None,
                token_pattern=self.token_pattern,
                show_progress=verbose,
            )
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens, show_progress=verbose)
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
        filters: Optional[MetadataFilters] = None,
        # deprecated
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Retriever":
        if tokenizer is not None:
            logger.warning(
                "The tokenizer parameter is deprecated and will be removed in a future release. "
                "Use a stemmer from PyStemmer instead."
            )

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            stemmer=stemmer,
            language=language,
            similarity_top_k=similarity_top_k,
            filters=filters,
            verbose=verbose,
            skip_stemming=skip_stemming,
            token_pattern=token_pattern,
        )

    def get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        persist_dict: Dict[str, Any] = {}
        for key in DEFAULT_PERSIST_ARGS:
            if not hasattr(self, key):
                continue
            val = getattr(self, key)
            if key == "filters" and val is not None:
                persist_dict[DEFAULT_PERSIST_ARGS[key]] = val.model_dump()
            else:
                persist_dict[DEFAULT_PERSIST_ARGS[key]] = val
        return persist_dict

    def persist(self, path: str, encoding: str = "utf-8", **kwargs: Any) -> None:
        """Persist the retriever to a directory."""
        self.bm25.save(path, corpus=self.corpus, **kwargs)
        with open(
            os.path.join(path, DEFAULT_PERSIST_FILENAME), "w", encoding=encoding
        ) as f:
            json.dump(self.get_persist_args(), f, indent=2)

    @classmethod
    def from_persist_dir(
        cls, path: str, encoding: str = "utf-8", **kwargs: Any
    ) -> "BM25Retriever":
        """Load the retriever from a directory."""
        bm25 = bm25s.BM25.load(path, load_corpus=True, **kwargs)
        with open(os.path.join(path, DEFAULT_PERSIST_FILENAME), encoding=encoding) as f:
            retriever_data = json.load(f)
        if "filters" in retriever_data and retriever_data["filters"] is not None:
            retriever_data["filters"] = MetadataFilters.model_validate(
                retriever_data["filters"]
            )
        return cls(existing_bm25=bm25, **retriever_data)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = bm25s.tokenize(
            query,
            stemmer=self.stemmer if not self.skip_stemming else None,
            token_pattern=self.token_pattern,
            show_progress=self._verbose,
        )
        weight_mask = None
        filter_fn = None
        if self._filters is not None:
            filter_fn = _build_metadata_filter_fn(
                lambda idx: self.corpus[idx], self._filters
            )
            mask = [1.0 if filter_fn(i) else 0.0 for i in range(len(self.corpus))]
            weight_mask = np.array(mask)

        indexes, scores = self.bm25.retrieve(
            tokenized_query,
            k=self.similarity_top_k,
            show_progress=self._verbose,
            weight_mask=weight_mask,
        )

        # batched, but only one query
        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
                idx_val = int(node.node_id)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
                idx_val = int(idx)

            if filter_fn is not None and not filter_fn(idx_val):
                continue

            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes

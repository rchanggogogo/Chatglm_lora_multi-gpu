# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/12 2:20 PM
==================================="""
import torch
from langchain.llms.base import LLM
from langchain.schema import Document
from pydantic import Extra

from typing import Any, List, Mapping, Optional
from langchain.vectorstores.milvus import Milvus


class CustomChatGLM(LLM):
    model_id = ''
    model_kwargs = {}
    model = Any
    tokenizer = Any
    device = ''


    class Config:
        extra = Extra.forbid

    def __init__(self, model_, tokenizer_, *args, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = {"_name_or_path": "/8t/workspace/lchang/models/chatglm-6B"}
        self.model_id = self.model_kwargs['_name_or_path']
        self.model = model_
        self.tokenizer = tokenizer_
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, *args, **kwargs) -> str:
        max_length = kwargs.get("max_length", 2048)
        temperature = kwargs.get("temperature", 0.8)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        out = self.model.generate(input_ids=input_ids, max_length=max_length, temperature=temperature)
        answer = self.tokenizer.decode(out[0])
        answer = answer[len(prompt):]
        if stop is not None:
            for s in stop:
                if s in answer:
                    answer = answer.split(s)[0]
                    break

        return answer

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**{"model": self.model_id}, **{"model_kwargs": self.model_kwargs}}


class CustomMilvus(Milvus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def similarity_search(
            self,
            query: str,
            k: int = 1,
            param: Optional[dict] = None,
            expr: Optional[str] = None,
            partition_names: Optional[List[str]] = None,
            round_decimal: int = -1,
            timeout: Optional[int] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """ custom myself similarity search

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 4.
            param (dict, optional): The search params for the index type.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): What partitions to search.
                Defaults to None.
            round_decimal (int, optional): What decimal point to round to.
                Defaults to -1.
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.

        Returns:
            List[Document]: Document results for search.

        """
        _, docs_and_scores = self._worker_search(
            query, k, param, expr, partition_names, round_decimal, timeout, **kwargs
        )
        threshold = kwargs.get('threshold', 0.0)
        final_result = []
        for document, score, _ in docs_and_scores:
            if score < threshold:
                continue
            meta_ = document.metadata
            if 'content_' in meta_:
                final_result.append(Document(page_content=meta_['content_'], metadata=meta_))
            else:
                final_result.append(document)
        return final_result

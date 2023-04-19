# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/19 5:09 PM
==================================="""
from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel, BaseRetriever, Document, BaseMessage, HumanMessage, AIMessage
from langchain.vectorstores.base import VectorStore


def _get_chat_history(chat_history: List[BaseMessage]) -> str:
    buffer = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            buffer += "\n" + "用户: " + message.content
        elif isinstance(message, AIMessage):
            buffer += "\n" + "system: " + message.content
    return buffer


class CustomConversationalRetrievalChain(Chain, BaseModel):
    """Chain for chatting with an index."""

    combine_docs_chain: BaseCombineDocumentsChain
    question_generator: LLMChain
    output_key: str = "answer"
    return_source_documents: bool = False
    get_chat_history: Optional[Callable[[Tuple[str, str]], str]] = None
    """Return the source documents."""
    retriever: BaseRetriever
    """Index to connect to."""
    max_tokens_limit: Optional[int] = None
    """If set, restricts the docs to return from store based on tokens, enforced only
    for StuffDocumentChain"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
                self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(self.memory.buffer)

        docs = self._get_docs(question, inputs)
        new_inputs = inputs.copy()
        new_inputs["question"] = question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.combine_docs_chain.combine_docs(docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


    async def _aget_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        """Get docs."""
        docs = await self.retriever.aget_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        docs = await self._aget_docs(new_question, inputs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = await self.combine_docs_chain.acombine_docs(docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not savable when `get_chat_history` is not None.")
        super().save(file_path)

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            retriever: BaseRetriever,
            condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
            qa_prompt: Optional[BasePromptTemplate] = None,
            chain_type: str = "stuff",
            **kwargs: Any,
    ) -> CustomConversationalRetrievalChain:
        """Load chain from LLM."""
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            prompt=qa_prompt,
        )
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )

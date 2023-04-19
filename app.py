from typing import List, Optional, Dict, Any, Tuple

import torch
from fastapi import FastAPI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
import requests

import os
from collections import deque
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.milvus import Milvus
from langchain.chains.base import Chain
from transformers import AutoTokenizer

from custom_chat_glm import CustomChatGLM, CustomMilvus
from custom_conversation_chain import CustomConversationalRetrievalChain
from modeling_chatglm import ChatGLMForConditionalGeneration
from parsing_utils import load_text_from_file
from langchain.chains import ConversationalRetrievalChain

device = "cuda:0" if torch.cuda.is_available() else "cpu"

embedding_func = HuggingFaceEmbeddings(
    model_name="/8t/workspace/lchang/models/paraphrase-multilingual-mpnet-base-v2")

tokenizer = AutoTokenizer.from_pretrained("/8t/workspace/lchang/models/chatglm-6B-new/chatglm-6b",
                                          trust_remote_code=True,
                                          device=device)
model = ChatGLMForConditionalGeneration.from_pretrained("/8t/workspace/lchang/models/chatglm-6B-new/chatglm-6b",
                                                        trust_remote_code=True).half().to(device)

# embedding_func = OpenAIEmbeddings()
# build chatglm llm
llm = CustomChatGLM(model_=model, tokenizer_=tokenizer, device=device)

# init milvus collection
texts, metadatas = load_text_from_file('/8t/workspace/lchang/models/data/only_jojo_with_instruction.txt')

# retrieval
vectordb = CustomMilvus.from_texts(texts=texts, embedding=embedding_func,
                                   connection_args={'host': '10.128.6.19', 'port': 19530}, metadatas=metadatas)

prompt_str = """叫叫APP是一个儿童成长数字内容综合平台。能够发掘每个孩子的内驱原动力，科学搭建成长梯子，助力孩子自我成长。叫叫目前拥有阅读、益智、美育、小作家等不同产品，满足孩子多样化的学习需求。
你是叫叫的客服机器人，你的工作是根据文档和历史对话中的信息，回答用户的问题。请注意，你的回答将会被用户评价，如果回答不符合要求，你的工作将会受到影响。
{documents}
---------------
历史对话：
{chat_history}
---------------
用户：{question}
system：
"""

stuff_prompt = PromptTemplate(
    template="文档：\n{page_content}\n",
    input_variables=["page_content"],
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,
                                  input_key="question",
                                  output_key='answer')

prompt_template = PromptTemplate(template=prompt_str, input_variables=["documents", "chat_history", "question"])

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

stuff_document_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="documents",
                                           document_prompt=stuff_prompt)

chat_history = []

conversation_retrieval_chain = CustomConversationalRetrievalChain(combine_docs_chain=stuff_document_chain,
                                                                  question_generator=llm_chain,
                                                                  output_key="answer",
                                                                  retriever=vectordb.as_retriever(),
                                                                  memory=memory
                                                                  )


def reset_memory(memory):
    memory.clear()
    return memory


class Item(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[dict] = []


app = FastAPI()


@app.post("/chatglm")
async def chatglm(item: Item):
    return {"message": "Hello World"}


@app.post("/chatgpt")
async def chatgpt(item: Item):
    return {"message": "Hello World"}


if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8091)
    answer1 = conversation_retrieval_chain(
        {'question': '叫叫的理念是什么？', 'chat_history': chat_history, 'stop': '\n\n'})
    print(answer1)

    print(conversation_retrieval_chain(
        {'question': '怎么才能变得更聪明？', 'chat_history': answer1['chat_history'], 'stop': '\n\n'}))

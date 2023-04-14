# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/7 10:12 AM
==================================="""
import os
from typing import Optional, List, Mapping, Any, Tuple

from langchain import PromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from custom_chat_glm import CustomChatGLM, CustomMilvus
from tqdm import tqdm
import gradio as gr

device = "cuda:3" if torch.cuda.is_available() else "cpu"


embedding_func = HuggingFaceEmbeddings(
    model_name="/8t/workspace/lchang/models/paraphrase-multilingual-mpnet-base-v2")

tokenizer = AutoTokenizer.from_pretrained("/8t/workspace/lchang/models/chatglm-6B-new/chatglm-6b", trust_remote_code=True,
                                          device=device)
model = ChatGLMForConditionalGeneration.from_pretrained("/8t/workspace/lchang/models/chatglm-6B-new/chatglm-6b",
                                                        trust_remote_code=True).half().to(device)


# build milvus client
# milvus_client = Milvus(embedding_function=embedding_func,
#                        collection_name='jojo_knowledge',
#                        connection_args={'host': '10.128.6.19', 'port': 19530},
#                        text_field='search_text')

# build chatglm llm
llm = CustomChatGLM(model_=model, tokenizer_=tokenizer, device=device)



def load_text_from_file(file_path, split_by='<i_am_split>'):
    texts = []
    metadatas = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='load text from file'):
                if split_by in line:
                    texts.append(line.split(split_by)[0].strip())
                    metadatas.append({'source': file_path.split('/')[-1],
                                      'content_': line.strip().replace(split_by, '')})
                else:
                    texts.append(line.strip())
                    metadatas.append({'source': file_path.split('/')[-1], 'content_': line.strip()})
    return texts, metadatas

# init milvus collection
texts, metadatas = load_text_from_file('/8t/workspace/lchang/models/data/only_jojo_with_instruction.txt')


# retrieval
vectordb = CustomMilvus.from_texts(texts=texts, embedding=embedding_func,
                             connection_args={'host': '10.128.6.19', 'port': 19530}, metadatas=metadatas)


# custom prompt
combined_prompt = """学习文本中的信息，回答用户的问题，如果你不知道答案，就说不知道，不要编造答案，不要回答无关内容。请用中文回答问题。
----
{summaries}
----
用户：{question}
system：
"""
# chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
#                                                     chain_type='stuff',
#                                                     retriever=vectordb.as_retriever(),
#                                                     max_tokens_limit=2048)

prompt = PromptTemplate(
    template=combined_prompt, input_variables=["summaries", "question"]
)
assistant_prompt_template = """叫叫APP是一个儿童成长数字内容综合平台。能够发掘每个孩子的内驱原动力，科学搭建成长梯子，助力孩子自我成长。叫叫目前拥有阅读、益智、美育、小作家等不同产品，满足孩子多样化的学习需求。
你是叫叫的客服助手，用来帮助回答用户关于叫叫的问题。作为客服助手能够根据用户的输入生成与之相关的回答，并且能够根据上下文给出符合语境的回答。
同时客服助手也需要对自己给出的回答进行评估，根据评估结果对自己的回答进行改进，提供更好的服务。
请注意，你的回答将会被用户评价，如果回答不符合要求，你的工作将会受到影响。

{history}
用户：{human_input}
system：
"""

assistant_prompt = PromptTemplate(
    template=assistant_prompt_template, input_variables=["history", "human_input"]
)
stuff = PromptTemplate(
    template="内容: {page_content}\n",
    input_variables=["page_content"],
)


# stuff_document_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="summaries",
#                                            document_prompt=stuff)
# chain = RetrievalQAWithSourcesChain(combine_documents_chain=stuff_document_chain, retriever=vectordb.as_retriever())
# # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, combine_prompt=prompt,question_prompt=question_prompt, retriever=vectordb.as_retriever(),
# #                                              max_tokens_limit=2048)
# # chain = load_qa_with_sources_chain(llm=llm, chain_type='stuff', retriever=vectordb.as_retriever())
# # index_creator = VectorstoreIndexCreator(embeddings=embeddings)
# # index = index_creator.from_loaders(loaders=[loader])
#

def load_chain(llm_):
    memory = ConversationBufferMemory(memory_key="history")
    # chains
    llm_chain = LLMChain(llm=llm, prompt=assistant_prompt, verbose=False, memory=memory)

    qa_chain = load_qa_with_sources_chain(llm=llm, chain_type='stuff', prompt=prompt, document_prompt=stuff, verbose=False)
    return qa_chain, llm_chain, memory


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, memory


class ChatWrapper:
    def __init__(self, llm_chain, qa_chain, memory, vectordb):
        self.llm_chain = llm_chain
        self.qa_chain = qa_chain
        self.memory = memory
        self.history = []
        self.vectordb = vectordb

    def __call__(self, human_input, history: Optional[Tuple[str, str]]):
        param = {"metric_type": "IP", "params": {"nprobe": 256}}
        docs = self.vectordb.similarity_search(human_input, k=1, param=param,threshold=0.8)
        final_result = {}
        if len(docs) > 0:
            ans = self.qa_chain({'input_documents': docs, 'question': human_input})
            final_result.update(ans)
            final_result['chat_message'] = ans['output_text']
        else:
            ans = self.llm_chain.predict(human_input=human_input)
            final_result['chat_message'] = ans
        self.history.append((human_input, final_result['chat_message']))
        return self.history, final_result

    def reset(self):
        self.history, self.memory = reset_memory(self.history, self.memory)


qa_chain, llm_chain, memory = load_chain(llm)

chat = ChatWrapper(llm_chain, qa_chain, memory, vectordb)

with gr.Blocks(css=".gradio-container {background-color: lightgray}") as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    memory_state = gr.State()
    llm_chain_state = gr.State()

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                gr.HTML("""<h1>叫叫客服助手</h1>""")

        with gr.Row():
            with gr.Column(scale=7, visible=True):
                chatbot = gr.Chatbot().style(height=450)

        with gr.Row():
            message = gr.Textbox(label="请输入你的问题",
                                 placeholder="叫叫的理念是什么？",
                                 lines=1)

            submit = gr.Button(value="发送", variant="secondary").style(full_width=False)

        gr.Examples(
            examples=["叫叫的理念是什么？",
                      "叫叫的四大理念是什么？",
                      "线上阅读内容会有老师指导吗？",
                      "要学多久才能看到效果？",
                      "如何开启阅读？",
                      "如何绑定亲情号？",
                      "在手机上学太伤害眼睛，不想给孩子看电子产品",
                      "成长豆兑换实物存在质量问题应如何处理？"],
            inputs=message
        )
    gr.HTML("""
            <p> 这是一个基于大模型的对话客服，用来为用户提供对有关叫叫APP问题的解答。由成都书声科技有限公司开发维护。 </p>
            <p> 注：免责声明</p>
            """)

    message.submit(chat, inputs=[message, history_state],
                   outputs=[chatbot, history_state])
    submit.click(chat, inputs=[message, history_state],
                 outputs=[chatbot, history_state])

block.launch(server_port=8090, server_name='0.0.0.0', debug=True)

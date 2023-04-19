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
from parsing_utils import load_text_from_file

device = "cuda:3" if torch.cuda.is_available() else "cpu"

embedding_func = HuggingFaceEmbeddings(
    model_name="/8t/workspace/lchang/models/paraphrase-multilingual-mpnet-base-v2")

tokenizer = AutoTokenizer.from_pretrained("/8t/workspace/lchang/models/chatglm-6B-new/chatglm-6b",
                                          trust_remote_code=True,
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

    qa_chain = load_qa_with_sources_chain(llm=llm, chain_type='stuff', prompt=prompt, document_prompt=stuff,
                                          verbose=False)
    return qa_chain, llm_chain, memory


def update_foo(widget, state):
    if widget:
        state = widget
        return state


def reset_memory(history):
    memory.clear()
    history = []
    return history, history, memory


def add_memory(file_path):
    texts, metadatas = load_text_from_file(file_path)
    try:
        vectordb.add_texts(texts, metadatas)
    except Exception as e:
        print(e)
        return 'Add failed'

    return 'Add success'


class ChatWrapper:
    def __init__(self, llm_chain, qa_chain, memory, vectordb):
        self.llm_chain = llm_chain
        self.qa_chain = qa_chain
        self.memory = memory
        self.vectordb = vectordb

    def __call__(self, human_input, history: Optional[Tuple[str, str]], top_p, temperature, **kwargs):
        kwargs['top_p'] = top_p
        kwargs['temperature'] = temperature
        k = kwargs.pop('k', 1)
        threshold = kwargs.pop('threshold', 0.8)
        param = {"metric_type": "IP", "params": {"nprobe": 256}}

        docs = self.vectordb.similarity_search(human_input, k=k, param=param, threshold=threshold)
        final_result = {}
        history = history or []
        if len(docs) > 0:
            ans = self.qa_chain({'input_documents': docs, 'question': human_input, **kwargs})
            final_result.update(ans)
            final_result['chat_message'] = ans['output_text']
        else:
            ans = self.llm_chain.generate([{'human_input': human_input,
                                            'history': history,
                                            'memory': self.memory,
                                            **kwargs}])
            if len(ans.generations) > 0:
                result = ans.generations[0][0].text
            else:
                result = '很抱歉，我无法回答你的问题。'
            final_result['chat_message'] = result
        history.append((human_input, final_result['chat_message']))
        return history, history, final_result


qa_chain, llm_chain, memory = load_chain(llm)

chat = ChatWrapper(llm_chain, qa_chain, memory, vectordb)

with gr.Blocks(css=".gradio-container {background-color: lightgray}") as block:
    history_state = gr.State()
    memory_state = gr.State()

    # generate state
    top_p_state = gr.State(0.9)
    temperature_state = gr.State(0.85)
    k_state = gr.State(1)
    threshold_state = gr.State(0.8)

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
            clear_submit = gr.Button(value="清空", variant="secondary").style(full_width=False)

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

    with gr.Tab("Settings"):
        top_p_slider = gr.Slider(label="top_p", maximum=1, step=0.01, value=0.9)
        temperature_slider = gr.Slider(label="temperature", maximum=1, step=0.01, value=0.85)
        k_slider = gr.Slider(label="k", mininum=1, maximum=10, step=1, value=1)
        threshold_slider = gr.Slider(label="threshold", maximum=1, step=0.1, value=0.8)
        top_p_slider.change(update_foo,
                            inputs=[top_p_slider, top_p_state],
                            outputs=[top_p_state])
        temperature_slider.change(update_foo,
                                  inputs=[temperature_slider, temperature_state],
                                  outputs=[temperature_state])
        k_slider.change(update_foo,
                        inputs=[k_slider, k_state],
                        outputs=[k_state])
        threshold_slider.change(update_foo,
                                inputs=[threshold_slider, threshold_state],
                                outputs=[threshold_state])

    with gr.Tab("Memory"):
        with gr.Row():
            upload_file = gr.File(label="上传文件，请使用'<i_am_split>'分割问题和答案在同一行内", type="file")
            upload_button = gr.Button(value="上传", variant="secondary").style(full_width=False)

    gr.HTML("""
            <p> 这是一个基于大模型的对话客服，用来为用户提供对有关叫叫APP问题的解答。由成都书声科技有限公司开发维护。 </p>
            <p> 注：免责声明</p>
            """)
    # use enter to submit
    message.submit(chat, inputs=[message, history_state, top_p_state, temperature_state],
                   outputs=[chatbot, history_state], show_progress=True, queue=False)
    submit.click(chat, inputs=[message, history_state, top_p_state, temperature_state],
                 outputs=[chatbot, history_state])
    clear_submit.click(reset_memory, inputs=[history_state],
                       outputs=[chatbot, history_state, memory_state])
    upload_button.click(add_memory, inputs=[upload_file],
                        outputs=[upload_button])

block.launch(server_port=8090, server_name='0.0.0.0', debug=True)

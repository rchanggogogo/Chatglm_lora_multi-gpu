# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/7 2:34 PM
==================================="""
import requests
import json

import gradio as gr
from typing import Optional, List, Mapping, Any, Tuple

# 设置环境，CPU还是GPU
history = []


def update_foo(widget, state):
    if widget:
        state = widget
        return state


def reset_memory(history):

    history = []
    chat.clear_history()
    return history, history



class ChatWrapper:
    def __init__(self):
        self.token = "24.1907b8e675f53a07c84a0890dcbc6ba8.2592000.1686107388.282335-32879611"
        self.history_ = []
        self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={}"
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def _refresh_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?client_id=q60lV0lWR7FLNLcubXnfOkTh&client_secret=wGghq4VykFvV8yoPlyP1ZHOu6daINbN0&grant_type=client_credentials"

        payload = ""
        response = requests.request("POST", url, headers=self.headers, data=payload)
        self.token = response.json()['access_token']

    def __call__(self, human_input, history: Optional[Tuple[str, str]], **kwargs):
        history = history or []
        if human_input.strip() == '':
            history.append((human_input, '请说点什么吧'))
            return history, history, '请说点什么吧'

        url = self.url.format(self.token)
        user_dict = {'role': 'user', 'content': human_input}
        assistant_dict = {'role': 'assistant', 'content': ''}
        # 模型生成

        if self.history_:
            if len(self.history_) > 20:
                self.history_.pop(0)
                self.history_.pop(0)
                self.history_.append(user_dict)
                payload = {
                    'messages': self.history_
                }
            else:
                self.history_.append(user_dict)
                payload = {
                    'messages': self.history_
                }
        else:
            self.history_.append(user_dict)
            payload = {
                'messages': self.history_
            }

        response = requests.post(url, data=json.dumps(payload), headers=self.headers)
        if response.status_code == 200:
            print(response.json())
            result = response.json()
            if 'error_code' in result:
                if result['error_code'] == 111:
                    self._refresh_token()
                    url = url.format(self.token)
                    response = requests.post(url, data=json.dumps(payload), headers=self.headers)
                    assistant_dict['content'] = response.json()['result']
                    self.history_.append(assistant_dict)
                    history.append((human_input, result['result']))
                    return history, history, result['result']
                else :
                    history.append((human_input, result['result']))
                    return history, history, result['error_msg']

            assistant_dict['content'] = result['result']
            self.history_.append(assistant_dict)
            history.append((human_input, result['result']))
            return history, history, result['result']

    def clear_history(self):
        self.history_ = []

chat = ChatWrapper()

with gr.Blocks(css=".gradio-container {background-color: lightgray}") as block:
    history_state = gr.State()
    memory_state = gr.State()

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                gr.HTML("""<h1>Erine 小能手</h1>""")

        with gr.Row():
            with gr.Column(scale=7, visible=True):
                chatbot = gr.Chatbot().style(height=450)

        with gr.Row():
            message = gr.Textbox(label="请输入你的问题",
                                 placeholder="讲个笑话",
                                 lines=1)

            submit = gr.Button(value="发送", variant="secondary").style(full_width=False)
            clear_submit = gr.Button(value="清空", variant="secondary").style(full_width=False)

        gr.Examples(
            examples=["你是谁",
                      "地球是圆的还是方的",
                      "给我写一篇关于小米手机10的文章"],
            inputs=message
        )


    gr.HTML("""
            <p> 这是一个大模型，用来提供对话体验服务。由成都书声科技有限公司开发维护。 </p>
            <p> 注：免责声明</p>
            """)
    # use enter to submit
    message.submit(chat, inputs=[message, history_state],
                   outputs=[chatbot, history_state], show_progress=True, queue=False)
    submit.click(chat, inputs=[message, history_state],
                 outputs=[chatbot, history_state])
    clear_submit.click(reset_memory, inputs=[history_state],
                       outputs=[chatbot, history_state])


block.launch(server_port=8093, server_name='0.0.0.0', debug=True)

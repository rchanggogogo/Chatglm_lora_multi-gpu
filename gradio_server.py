# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/4/7 2:34 PM
==================================="""
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import gradio as gr

# 设置环境，CPU还是GPU
torch.set_default_tensor_type(torch.cuda.HalfTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# 加载base model
model = ChatGLMForConditionalGeneration.from_pretrained("/8t/workspace/lchang/models/chatglm-6B",
                                                        trust_remote_code=True).half().to(device)

# 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained("/8t/workspace/lchang/models/chatglm-6B", trust_remote_code=True)

# 加载训练好的lora
peft_path = "/8t/workspace/lchang/models/ChatGLMJOJO_one/saved/finetune_19.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def chatbot(text):
    # 模型生成
    input_ids = tokenizer.encode(text, return_tensors='pt')
    # model.cuda()
    out = model.generate(input_ids=input_ids, max_length=2048, temperature=0.8)
    answer = tokenizer.decode(out[0])
    return answer


iface = gr.Interface(chatbot, inputs="textbox", title="JoJo Chatbot", description="Chatbot for JOJO", outputs="text")
iface.launch(share=True, debug=True, server_port=8087, server_name="0.0.0.0")
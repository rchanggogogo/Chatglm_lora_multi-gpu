from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

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
peft_path = "/8t/workspace/lchang/models/ChatGLMJOJO_one/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# 模型生成
text = "<|im_start|>你是由叫叫训练的对话模型，用来为叫叫的用户提供对话服务，解答有关叫叫的相关问题。\n<|im_end|><|im_start|>叫叫的理念是什么？<|im_end|>"
input_ids = tokenizer.encode(text, return_tensors='pt')
# model.cuda()
out = model.generate(input_ids=input_ids, max_length=2048, temperature=0)
answer = tokenizer.decode(out[0])
print(answer)

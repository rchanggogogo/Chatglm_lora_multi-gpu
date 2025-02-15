# Chatglm_lora_multi-gpu
chatglm多gpu用deepspeed和
## 大模型prompt&delta理论部分知识 ##
1.**[CSDN链接](https://mp.csdn.net/mp_blog/creation/editor/129835450)**

2.**[知乎链接](https://zhuanlan.zhihu.com/p/617919855)**
### 迭代比较匆忙，空了我会重新整理 ###
## 初始化环境 ##
<code>pip install -r requirements.txt</code>
## 包括3种方式多gpu运行： ##
### 0 最简单的多gpu运行，能跑通的单机脚本+deepspeed的配置文件就可以 ###
<div>
  单机执行命令
<code>
  python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 2000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --report_to wandb
    --output_dir output
  </code>
</div>

<div>
  多Gpu执行命令
  <code>
    torchrun --nproc_per_node=2 multi_gpu_fintune_belle.py  \
             --dataset_path data/alpaca  \
             --lora_rank 8 \
             --per_device_train_batch_size 1 \
             --gradient_accumulation_steps 1 \
              --save_steps 2000 \
              --save_total_limit 2 \
              --learning_rate 2e-5 \
              --fp16 \
              --num_train_epochs 2 \
              --remove_unused_columns false \
              --logging_steps 50 \
              --report_to wandb
              --output_dir output \
              --deepspeed ds_config_zero3.json
  </code>
</div>

### 1.deepspeed ###
#### 数据处理 ####
<div>给两份belle中文的self instruct数据

         1.0.5M版本：
          cd data 
          
          wget https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN/resolve/main/Belle.train.json

         2.1M版本
         
         wget https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN/resolve/main/belle_open_source_1M.train.json

         3.把两份数据合并成一份

         a.0.5M和1M数据字段有些不同，统一处理数据，用地下代码处理1M数据
         
         cd ..
         
         python process_belle_1M_data.py

         b.把两份文件合并成一份，命名为：Belle_0_1.train.json

         cd data & cat Belle.train.json Belle_1M.train.json>Belle_0_1.train.json
</div>

#### 数据准备好后执行下面命令 ####
<code>torchrun --nproc_per_node=2 multi_gpu_fintune_belle.py \\
         --dataset_path data/alpaca \\
         --lora_rank 8 \\
         --per_device_train_batch_size 1 \\
         --gradient_accumulation_steps 1 \\
         --save_steps 1000 \\
         --save_total_limit 2 \\
         --learning_rate 2e-5 \\
         --fp16 \\
         --num_train_epochs 2 \\
         --remove_unused_columns false \\
         --logging_steps 50 \\
         --gradient_accumulation_steps 2 \\
         --output_dir output \\
         --deepspeed ds_config_zero3.json
</code>


### 2.accelerate+deepspeed ### 

#### 准备数据 ####
<div>

下载数据

cd data 
          
wget https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN/resolve/main/Belle.train.json

<code>python tokenize_dataset_rows_belle.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 200 \
    --skip_overlength
</code>
</div>

#### 数据准备好后执行下面命令 ####
<code>accelerate launch --config_file accelerate_ds_zero3_cpu_offload_config.yaml  multi_gpu_fintune_belle.py \\
                  --dataset_path data/alpaca  \\
                  --lora_rank 8 \\
                  --per_device_train_batch_size 2 \\
                  --gradient_accumulation_steps 1 \\
                  --max_steps 10000 \\
                  --save_steps 1000 \\
                  --save_total_limit 2 \\
                  --learning_rate 2e-5 \\
                  --fp16 \\
                  --remove_unused_columns false \\
                  --logging_steps 50 \\
                  --output_dir output
</code>

### 3.ddp方式还没试 ###

## batch inference ##

实际工作中常常会出现，需要批量不数据预测出来问题

往往我们有一台高性能的机器，但是如果做fintune，只能一张卡一个时间面对一个请求，造成显卡存资源浪费

batch inference成为必要

1.<code>deepspeed --num_gpus 2 chatglm_deepspeed_inference.py</code>

2.显卡资源不足以装下大模型，可以用accelerate.load_checkpoint_and_dispatch：

<div>python chatglm_milti_gpu_inference.py</code>

如果也想用deepspeed加速，把以下注释代码去掉：
<div>
         <code># init deepspeed inference engine
'''ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=8,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
                  print(f"model is loaded on device {ds_model.module.device}")'''</code>
         
</div>
<code>deepspeed --num_gpus 2 chatglm_milti_gpu_inference.py</code>

## webUI交互 ##
进入webui文件夹，执行readme.txt命令即可
![image](https://user-images.githubusercontent.com/9170648/229347851-e4047f85-4ab9-4ba2-bbb2-219375d40465.png)


<code> streamlit run web_feedback.py --server.port 6006 </code>

## 新增chatglm作图应用 ##
![生成图](https://user-images.githubusercontent.com/9170648/229387760-b72b063a-5cd2-4243-b204-b4f782692d9b.png)

进入APP——example应用

![023106E2-912D-4999-A0A2-9971C36A0769](https://user-images.githubusercontent.com/9170648/229387734-9a4c3c88-50ae-4492-b897-aba20f9cb46e.png)


![7762BA98-AE3C-4D28-8CFD-8531A1C9209A](https://user-images.githubusercontent.com/9170648/229387742-35616814-3b60-43c4-9b5b-94be7720f0ab.png)






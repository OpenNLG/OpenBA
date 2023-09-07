# OpenBT5-LM
An Open-Source Bilingual Language Model Based on the T5 Architecture.

## Training

Our training code are put in folder `training`. Based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/), we made the following implementations:
- SwiGLU activation function,
- UL2 training objective,
- Rotary positional embedding,
- A unified MMap data processing method for both pre-training and fine-tuning phases.

For pre-training, relevant requirements should be installed beforehand as stated in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/), then you can simply run the following command to process texts into bytes, which can be read faster by a MMap Dataset:

```bash
cd training
bash scripts/data_process_span_corr.sh  # process pretrain data
bash scripts/data_process_flan.sh  # process fine-tune data
```

The you can run distributed training across multi nodes by
```bash
bash scripts/run_pretrain.sh  # pre-train
bash scripts/run_stretch.sh  # length adaptation
bash scripts/run_flan.sh   # fine-tune
```

## Inference
To make it more convenient for developers to use, we have refactored our model to follow the design principles of the Huggingface Transformers Library.
You can simply 

安装依赖：
```bash
pip install transformers torch>=2.0 sentencepiece
```

OpenBT5-LM:
```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> tokenizer = AutoTokenizer.from_pretrained("OpenBT5/OpenBT5-LM", trust_remote_code=True)
>>> model = AutoModelForSeq2SeqLM.from_pretrained("OpenBT5/OpenBT5-LM", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> query = "<S>" + "苏州处太湖平原，沿江为高沙平原，河" + "<extra_id_0>"
>>> inputs = tokenizer(query, return_tensors="pt").to("cuda")
>>> outputs = model.generate(**inputs, do_sample=True, max_new_tokens=32)
>>> response = tokenizer.decode(outputs[0], skip_special_tokens=True)
>>> print(response)
流两侧为河淤平原,苏州平原是江苏平原主体,地势低平,土地肥沃,气候温和
```

OpenBT5-Flan:
```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> tokenizer = AutoTokenizer.from_pretrained("OpenBT5/OpenBT5-Flan", trust_remote_code=True)
>>> model = AutoModelForSeq2SeqLM.from_pretrained("OpenBT5/OpenBT5-Flan", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> query = "<S>" + "介绍一下中国的四大名著，并分别概括其主要内容" + "<extra_id_0>"
>>> inputs = tokenizer(query, return_tensors="pt").to("cuda")
>>> outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)
>>> response = tokenizer.decode(outputs[0], skip_special_tokens=True)
>>> print(response)
中国的四大名著分别是《红楼梦》、《西游记》、《水浒传》和《三国演义》。它们分别包括故事情节、文化内涵和历史背景等方面的不同特点。《红楼梦》是一部中国古典小说,讲述了贾宝玉、林黛玉、薛宝钗等一群人物在贾府的生活和爱情故事。《西游记》是中国著名小说,描述了孙悟空、猪八戒、沙悟净等一众妖魔鬼怪的冒险历程和故事。《水浒传》是一部中国古典小说,描述了宋江等一百零八位好汉的反抗故事。《三国演义》是中国古代著名小说,讲述了三国时期的历史和战争故事。这些小说在文学、历史、哲学和文化等方面都有着不同的影响和地位。
```

## Huggingface
We open-source two versions of our models:

- [OpenBT5-LM](https://huggingface.co/OpenBT5/OpenBT5-LM): 
- [OPenBT5-Flan](https://huggingface.co/OpenBT5/OpenBT5-Flan):
- OpenBT5-Chat: coming soon

import gradio as gr
import random
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import os

tokenizer = AutoTokenizer.from_pretrained('OpenBA/OpenBA-Flan', trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained('OpenBA/OpenBA-Flan', trust_remote_code=True).half().cuda()
model.eval()

def case_insensitive_replace(input_str, from_str, to_str):
    pattern = re.compile(re.escape(from_str), re.IGNORECASE)
    return pattern.sub(to_str, input_str)


def history2input(chat_history, message):
    input_text = ""
    for i, j in chat_history:
        input_text += f"Human: {i} </s> Assistant: {j} </s> "
    return input_text + f"Human: {message} </s> Assistant: "

def gpu_respond(message, top_p, temp, chat_history):
    input_text = history2input(chat_history, message)
    print("input:", input_text)
    bot_message = generate(input_text, top_p, temp)
    print("message:", bot_message)
    print('-' * 30)
    chat_history.append((message, bot_message))
    return "", chat_history

def generate(input_text, top_p = 0.7, temp = 0.95):
    inputs = tokenizer("<S> " + input_text + " <extra_id_0>", return_tensors='pt')
    for k in inputs:
        inputs[k] = inputs[k].cuda()

    outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=512,
                    temperature = temp,
                    top_p = top_p,              
                )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7,  label="Top P")
        temp = gr.Slider(minimum=0.01, maximum=1.0, value=0.95,  label="Temperature")

        msg.submit(gpu_respond, [msg, top_p, temp, chatbot], [msg, chatbot])

    demo.queue(concurrency_count=3)    
    demo.launch(share=True)

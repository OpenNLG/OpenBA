import re

def make_ABCD_input_0_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} Options: A. {:} B. {:} C. {:} D. {:} Answer:"
    ANS_TEMPLATE = ""
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"])
    decoder_input_text = ANS_TEMPLATE 
    return [input_text], decoder_input_text

def make_ABCD_input_5_shot(subject, data):
    demo = data['data']
    ASK_TEMPLATE = "Question: {:} Options: A. {:} B. {:} C. {:} D. {:} Answer:"
    ANS_TEMPLATE = "{:}"
    input_text = ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) # origin
    decoder_input_text = ANS_TEMPLATE 
    demos = data["demo"]
    fs_input_text = ""
    input_texts = [input_text]
    for demo in demos:
        fs_input_text += ASK_TEMPLATE.format(demo["question"], demo["res1"], demo["res2"], demo["res3"], demo["res4"]) + \
                         ANS_TEMPLATE.format(demo[f"ans"]) + '\n ' # origin
        input_texts.append(fs_input_text + input_text)
    return input_texts, decoder_input_text


def choose_longest_input(cand, max_length, tokenizer, add_s):
    idx = len(cand) - 1
    while idx >= 0:
        length = len(tokenizer(cand[idx])["input_ids"])
        if add_s: length += 2
        if length <= max_length:
            return cand[idx]
        idx -= 1
    return cand[0]




# MMLU evaluation

Here we provide scripts for inference of MMLU with OpenBA.

`make_data.py` is the script to construct `./data/5shot`, which is the MMLU dataset in json format.

```bash
mkdir ./output
bash scripts/eval_fewshot.sh # for few shot
bash scripts/eval_zeroshot.sh # for zero shot
```

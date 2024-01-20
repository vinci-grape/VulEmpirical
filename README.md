# VulEmpirical

### Dataset

The dataset is organized as follows:

[Big-Vul](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset)

After download and unzip data files, you should see the following directory structure:

```
VulEmpirical
├── dataset
    ├── Big-Vul
├── ...
```

### Scripts

1. How to Fine-tune LLMs (e.g., CodeLlama):
```
bash finetune.sh CodeLlama detection
```

2. How to prompt LLMs by Few-Shot learning:
```
python inference.py --model CodeLlama --task fewshot
```

After Fine-Tuning or Prompting, your directory structure should be like the following:
```
VulEmpirical
├── dataset
    ├── Big-Vul
├── fewshot
├── result
    ├── assessment
    ├── description
    ├── detection
    ├── location
output
    ├── CodeLlama
    ├── ...
├── ...
```
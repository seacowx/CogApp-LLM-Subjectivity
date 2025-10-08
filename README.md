# CogApp-LLM-Subjectivity
Official repository for the paper "Modeling Subjectivity in Cognitive Appraisal with Language Models"

Thank you for your interest in our project. We will upload the data and code for our experiments soon. Stay tuned!

## LLM Prompting Experiments (Zero-shot prompting from Section 4.2)
To run the experiments, use the `llm/exp.py` script.

### Arguments

*   `--model`: The name of the model to use. Choose from `llama8`, `qwen7`, `llama70`, and `qwen72`. (Required)
*   `--dataset`: The dataset to use. Choose from `envent`, `fge`, and `covidet`. (Default: `envent`)
*   `--add_demo`: Add demographic information to the prompt. (Only available for the `envent` dataset)
*   `--add_traits`: Add personality traits to the prompt. (Only available for the `envent` dataset)

### Example

```bash
python llm/exp.py --model llama8 --dataset envent --add_demo --add_traits
```

### Evaluating LLM Peformance
To evaluate the performance of the LLM, use the `llm/evaluate.py` script.

#### Arguments

*   `--dataset`: The dataset to use. Choose from `envent`, `fge`, and `covidet`. (Default: `envent`)
*   `--baseline`: The baseline to use. Choose from `random` and `majority`. Leave empty for LLM.
*   `--custom_fpath`: Custom file path to result files.
*   `--with_demo`: Evaluate on prompts with demographic information.
*   `--with_traits`: Evaluate on prompts with personality traits.
*   `--model_size`: The model size to use. Choose from `small` and `large`. (Default: `small`)

#### Example

```bash
python llm/evaluate.py --dataset envent --with_demo --with_traits --model_size small
```

## Label Smoothing Baseline (CASE-LSM from Section 4.1)
To train the label smoothing baseline model, use the `label_smoothing/train.py` script.

### Arguments

*   `--model`: The name of the transformer model to use. (Default: `microsoft/deberta-v3-large`)
*   `--modal`: The modality to use. `1` for unimodal, `2` for bimodal. (Default: `1`)
*   `--with_demo`: Use demographic information.
*   `--with_traits`: Use personality traits information.
*   `--state_dict_path`: The path to the folder for saving model weights.
*   `--resume_state_dict_path`: The path to a checkpoint to resume training from.

### Example

```bash
python label_smoothing/train.py --model microsoft/deberta-v3-large --modal 1 --with_demo --state_dict_path ./models
```

## Calibration Experiments (Post-hoc Calibration from Section 4.2)
To run the calibration experiments, use the `calibration/run.py` script.

### Arguments

*   `--model`: The name of the model to use. Choose from `llama8` and `qwen7`. (Required)
*   `--dataset`: The dataset to use. Choose from `envent`, `fge`, and `covidet`. (Default: `envent`)
*   `--add_demo`: Add demographic information to the prompt. (Only available for the `envent` dataset)
*   `--add_traits`: Add personality traits to the prompt. (Only available for the `envent` dataset)
*   `--eval_method`: The evaluation method to use. Choose from `consistency`, `avg-conf`, and `pair-rank`. (Default: `consistency`)

### Example

```bash
python calibration/run.py --model llama8 --dataset envent --eval_method consistency
```

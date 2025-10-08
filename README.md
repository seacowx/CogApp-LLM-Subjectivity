# CogApp-LLM-Subjectivity
Official repository for the paper "Modeling Subjectivity in Cognitive Appraisal with Language Models"

Thank you for your interest in our project. We will upload the data and code for our experiments soon. Stay tuned!

## Usage

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

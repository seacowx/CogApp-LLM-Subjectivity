**Procedure to run llm baselines**
```bash
python exp.py --model [MODEL]
```
where [MODEL] is one of the following:
- llama8
- qwen7

To add new model, navigate to `~/llm_baseline/llms.py` and the `vLLMInference` class. Append model by adding a new `elif` block that associates the `model_name` with the model path.
```python
elif model_name == 'new_model':
    model_path = 'ABSOLUTE PATH TO NEW MODEL'
```

**Adding auxiliary information**
To include auxiliary information such as the demographic information or personality traits, add the corresponding flag when running the experiment.
```bash
python exp.py --model llama8 --with_demo --with_traits
```
where:
- `--with_demo` includes demographic information
- `--with_traits` includes personality traits

Simply add both flags to include both demographic information and personality traits.
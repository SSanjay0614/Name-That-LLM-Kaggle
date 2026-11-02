# NTL V27 Qwen20 Atlas18

- **Competition:** Name That LLM (`name-that-llm`)
- **Parent:** V26 (`0.644`, submission `57831956`)
- **Change:** one blend-weight ablation at fixed ATLAS share: Qwen 0.18 -> 0.20 and Helper 0.12 -> 0.10; ATLAS remains 0.18.
- **Validation:** no paired prompt-grouped OOF is available for this tuple. The Kaggle fast-commit preview validates packaging only; the competition rerun is the accuracy evaluation.
- **Kaggle submission:** `57874985`, pending.

Build from the project root with `python .\name_that_llm\build_v27.py`, then push this folder to Kaggle.

# Tansformer-decoder-From-Scratch: Character-Level Transformer

A **character-level Transformer** trained on the **ChikSpir dataset** (~300,000 tokens) to generate Shakespeare-style text.  
The model contains approximately **10 million parameters** and learns to predict the next character given a context.

---

## Dataset

- **Source:** ChikSpir dataset (Shakespeare-style plays, public domain)  
- **Size:** ~1,000,000 characters (~300,000 tokens)
## Generating Text
- Model checkpoints are saved to checkpoints/final_model.pt.
- After training, you can generate text using a trained checkpoint.

## Notes

This project demonstrates a from-scratch character-level Transformer inspired by Andrej Karpathy’s minGPT.
Small model (~10M parameters) is suitable for educational purposes and can run on a single GPU or CPU.
The repo includes everything: dataset sample (input.txt), model code, training script, generation script, and saved weights (~52,000 KB).

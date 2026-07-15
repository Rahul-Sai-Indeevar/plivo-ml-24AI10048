# Training Run Log

**Run 1 (Baseline)**
* Hypothesis: Establish a baseline with the starter code to see how the byte-level tokenizer and default GPT perform.
* What changed: Unmodified starter code.
* Dev BPB: 2.35
* Conclusion: The score is poor. The byte-level tokenizer forces 3 tokens per Devanagari character, blowing up the sequence length and ruining parameter efficiency.

**Run 2 (Char Tokenizer + Mini LLaMA)**
* Hypothesis: A custom Character-Level Tokenizer with a UTF-8 byte fallback will compress Hindi text by 3x. Tying weights and using a Cosine schedule will maximize our 2000 step budget.
* What changed: Replaced tokenizer to map multi-byte chars to single IDs. Changed model to 4-layer / 192-dim with `tie_weights=True`. Added Cosine LR Schedule and AdamW.
* Dev BPB: 2.2246
* Conclusion: Massive improvement! However, I noticed the model is still underfitting slightly in the 2000 step limit due to dropout, and the context window is a bit small.

**Run 3 (Final Optimization)**
* Hypothesis: Because we are severely limited by 2000 steps, setting dropout to 0.0 will allow maximum feature learning. Increasing the block_size to 320 will give the evaluator a longer sliding window, lowering the BPB directly.
* What changed: Increased `block_size` to 320, increased `batch_size` to 24, set `dropout` to 0.0, and bumped max learning rate to 8e-4. 
* Dev BPB: 1.9707
* Conclusion: The winning run. The larger context window and zero-dropout allowed the model to over-index on the text structure rapidly, breaking the 2.0 BPB barrier while staying perfectly at the edge of the 2M parameter cap (1,989,312).

1. My best configuration pairs a custom character-level tokenizer with a tightly optimized 4-layer GPT architecture (1,989,312 parameters).
2. The core trick was addressing the Devanagari sequence bloat by dynamically assigning single vocabulary IDs to multi-byte UTF-8 sequences, compressing the Hindi text up to 3x.
3. I retained a 0-255 mapping for unknown bytes to ensure 100% lossless UTF-8 fallback.
4. To allow for a larger hidden dimension (192) without violating the 2,000,000 parameter limit, I enabled weight tying (`tie_weights=True`) which halves the vocabulary parameter footprint.
5. I removed bias tensors from LayerNorms and Linear layers, which trims unnecessary parameters so I could expand the context window to 320.
6. The batch size was pushed to 24 and dropout was dropped to 0.0, ensuring the model learned as aggressively as possible since we only have 2000 steps.
7. For the strict 2,000 step training limit, I implemented a Cosine Annealing Learning Rate Scheduler to avoid wasting optimization steps.
8. The scheduler incorporates a 10% step warmup phase to safely establish gradients before dynamically decaying the rate to force local minimum convergence by step 2,000.
9. I swapped the base Adam optimizer out for AdamW with a weight decay of 0.05 to improve generalization.
10. Finally, I added gradient clipping (max norm 1.0) to prevent training collapse from unstable gradients caused by the aggressive learning rate.
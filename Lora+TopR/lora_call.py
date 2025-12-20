"""
Entry point for running LoRA experiments with Top-R gradient masking.
Iterates through different combinations of ranks and top_r fractions
to evaluate their impact on model performance and parameter efficiency.
"""
from lora_config.lora import lora_run, MemoryPeakPerEpochCallback

num_train_epochs, learning_rate = 10, 2e-4
top_rs = [0.1, 0.3, 0.5]
ranks = [4,8,16,64,128]
datasets = ['sst2', 'imdb']
for dataset in datasets:
    for top_r in top_rs:
        for rank in ranks:
            runner = lora_run(num_train_epochs, rank, learning_rate, dataset, top_r=top_r)
            runner.run()

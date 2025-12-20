"""
Entry point for running LoRA (Low-Rank Adaptation) fine-tuning experiments.
Iterates through different dataset and rank configurations, performing both
model training and memory/FLOPs profiling.
"""
from lora_config.lora import lora_run, MemoryPeakPerEpochCallback
from forward_pass.profile_forward import profiler_forward as profile_forward

num_train_epochs, learning_rate = 10, 2e-4
ranks = [4,8,16,64,128]
datasets = ['sst2', 'imdb']
for dataset in datasets:
    for rank in ranks:
        runner = lora_run(num_train_epochs, rank, learning_rate,dataset)
        runner.run()
        profile_forward(rank,dataset).runner()    

from lora_config.lora import lora_run, MemoryPeakPerEpochCallback
from forward_pass.profile_forward import profiler_forward as profile_forward

num_train_epochs, learning_rate = 5, 2e-4
ranks = [4,8,16,64,128]
for rank in ranks:
    runner = lora_run(num_train_epochs, rank, learning_rate)
    runner.run()
    profile_forward(rank).runner()    
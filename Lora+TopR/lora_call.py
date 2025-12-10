from lora_config.lora import lora_run, MemoryPeakPerEpochCallback

num_train_epochs, learning_rate = 5, 2e-4
ranks = [8]
for rank in ranks:
    runner = lora_run(num_train_epochs, rank, learning_rate)
    runner.run()

from lora_config.lora import lora_run, MemoryPeakPerEpochCallback
num_train_epochs, rank, learning_rate = 5, 4, 2e-4
for i in range(3):
    runner = lora_run(num_train_epochs, rank, learning_rate)
    runner.run()
    rank = rank * 2
from lora_config.lora import lora_run, MemoryPeakPerEpochCallback

num_train_epochs, learning_rate = 10, 2e-4
ranks = [4,8,16,64,128]
datasets = ['sst2', 'imdb']
for dataset in datasets:
    for rank in ranks:
        runner = lora_run(num_train_epochs, rank, learning_rate,dataset)
        runner.run()

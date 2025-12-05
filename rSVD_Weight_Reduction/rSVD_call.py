from rSVD_HF_config.lora import lora_run
from forward_pass.profile import profile_forward as profile_forward

num_train_epochs, learning_rate = 5, 2e-4
ranks = [4,8,16,64,128]
for rank in ranks:
    runner = lora_run(num_train_epochs, rank, learning_rate)
    runner.run()
    profile_forward(rank).runner()  

from rSVD_HF_config.rSVD_finetuning import rSVD_run
from forward_pass_config.profile_forward import profiler_forward


num_train_epochs, learning_rate = 1, 2e-4
#ranks = [4,8,16,64,128]
ranks = [8]
for rank in ranks:
    runner = rSVD_run(num_train_epochs, rank, learning_rate)
    runner.run()
    profiler_forward(rank).runner()  

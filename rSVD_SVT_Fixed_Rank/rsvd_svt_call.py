"""
Entry point for rSVD SVT (Fixed Rank) fine-tuning experiments.
Coordinates the execution of training runs across multiple ranks,
leveraging the rSVDSVTAdam optimizer for parameter efficiency.
"""
from rsvd_svt_config.rsvd_svt import rsvd_svt_run
from forward_pass.profile_forward import profiler_forward as profile_forward

num_train_epochs, learning_rate = 10, 2e-4
ranks = [4,8,16,64,128]
dataset = "sst"  # or "imdb"
proj_interval = 500
use_rgp = True
gradient_accumulation_steps = 4

for rank in ranks:
    runner = rsvd_svt_run(
        num_train_epochs=num_train_epochs,
        rank = rank,
        learning_rate=learning_rate,
        dataset=dataset,
        proj_interval=proj_interval,
        use_rgp=use_rgp,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    runner.run()
    profile_forward(rank, dataset=dataset).runner()


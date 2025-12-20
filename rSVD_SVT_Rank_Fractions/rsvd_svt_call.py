"""
Entry point for rSVD SVT (Rank Fractions) fine-tuning experiments.
Coordinates the execution of training runs using different rank fractions,
leveraging the rSVDSVTAdam optimizer for parameter efficiency.
"""
from rsvd_svt_config.rsvd_svt import rsvd_svt_run
from forward_pass.profile_forward import profiler_forward as profile_forward

num_train_epochs, learning_rate = 10, 2e-4
rank_fractions = [0.01, 0.05, 0.1, 0.25, 0.5] 
dataset = "sst"  # or "imdb"
proj_interval = 500
use_rgp = True
gradient_accumulation_steps = 4

for rank_fraction in rank_fractions:
    runner = rsvd_svt_run(
        num_train_epochs=num_train_epochs,
        rank_fraction=rank_fraction,
        learning_rate=learning_rate,
        dataset=dataset,
        proj_interval=proj_interval,
        use_rgp=use_rgp,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    runner.run()
    profile_forward(rank_fraction, dataset=dataset).runner()


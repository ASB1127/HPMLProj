from rsvd_config.rsvd import rsvd_run
from forward_pass.profile_forward import profiler_forward as profile_forward

num_train_epochs, learning_rate = 10, 2e-4
rank_fractions = [0.01, 0.05, 0.1, 0.25, 0.5] 
dataset = "sst"  # or "imdb"
proj_interval = 500
use_rgp = True
gradient_accumulation_steps = 4

for rank_fraction in rank_fractions:
    runner = rsvd_run(
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


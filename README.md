# HPML Project: Efficient Fine-Tuning via Low-Rank, Sparse, and Subspace-Projected Gradient Updates

## Team Information
- **Team Name**: APA
- **Members**:
  - Amit Bal (asb2332)
  - Pranav Hariharane (pkh2120)
  - Athitheya Gobinathan (asg2278)

---

## 1. Problem Statement

This project investigates whether **low-rank structure, sparsity, and subspace projection** can reduce memory and compute requirements during fine-tuning without degrading convergence quality. We compare multiple parameter-efficient fine-tuning approaches on sentiment analysis tasks:

- **LoRA (Low-Rank Adaptation)**: Injects trainable low-rank matrices into transformer layers
- **rSVD (Randomized SVD)**: Uses randomized singular value decomposition for gradient compression
- **rSVD + SVT (Singular Value Thresholding)**: Combines rSVD with adaptive rank selection
- **rSVD with Weight Reduction**: Applies weight compression techniques
- **rSVD with Weight Reduction + SVT**: Combines rSVD with weight compression and adaptive rank selection

The goal is to achieve competitive accuracy on text classification benchmarks while dramatically reducing memory footprint and computational cost.

---

## 2. Model Description

### Base Model
- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Framework**: PyTorch with Hugging Face Transformers
- **Datasets**: 
  - SST-2 (Stanford Sentiment Treebank - binary sentiment classification)
  - IMDB (Movie reviews - binary sentiment classification)

### Fine-Tuning Approaches

#### LoRA Configuration
- Rank values tested: [4, 8, 16, 64, 128]
- Target modules: Query and Value projection layers
- Learning rate: 2e-4
- Training epochs: 10

#### rSVD Configuration
- Rank values tested: [4, 8, 16, 64, 128]
- Projection interval: 500 steps
- Uses Randomized Gradient Projection (RGP)
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Training epochs: 10

#### rSVD Variants
- **SVT Fixed Rank**: Fixed rank with singular value thresholding
- **SVT Rank Fractions**: Adaptive rank selection using fraction-based thresholds
- **Weight Reduction**: Applies compression to model weights
- **Weight Reduction + SVT**: Combines both techniques

---

## 3. Final Results Summary

### Memory Efficiency (Optimizer Memory Comparison)

| Method | Rank | Memory (MB) | Memory Savings | Savings % |
|--------|------|-------------|----------------|-----------|
| Adam (Baseline) | - | 510.83 MB | - | - |
| rSVDAdam | 4 | 2.92 MB | 507.91 MB | **99.43%** |
| rSVDAdam | 8 | 5.37 MB | 505.46 MB | **98.95%** |
| rSVDAdam | 16 | 10.25 MB | 500.58 MB | **97.99%** |
| rSVDAdam | 64 | 39.57 MB | 471.25 MB | **92.25%** |
| rSVDAdam | 128 | 78.67 MB | 432.15 MB | **84.60%** |

### Training Performance (SST-2, Rank 16)

| Method | Final Train Loss | Final Eval Loss | Peak Memory (GB) |
|--------|------------------|-----------------|------------------|
| LoRA | 0.1133 | 0.3944 | 1.40 GB |
| rSVD | 0.0727 | 0.3252 | 1.85 GB |

### Key Findings
- **rSVD achieves up to 99.43% optimizer memory savings** compared to standard Adam
- **Lower training loss** with rSVD (0.0727 vs 0.1133 for LoRA at rank 16)
- **Competitive evaluation performance** across all methods
- **Trade-off**: Memory savings vs. convergence speed varies by rank selection

---

## 4. Reproducibility Instructions

### A. Requirements

The project uses different requirements for different approaches. Install the base dependencies:

```bash
# For rSVD variants
cd rSVD
pip install -r requirements.txt

# Or for SVT variants
cd rSVD_SVT_Fixed_Rank
pip install -r requirements.txt
```

**Core Dependencies**:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `datasets>=2.12.0`
- `numpy>=1.24.0,<2.0.0`
- `tqdm>=4.65.0`

---

### B. Training

Each approach has its own training script. Navigate to the desired directory and run:

#### LoRA Training
```bash
cd Lora
python lora_call.py
```

This will train LoRA models with ranks [4, 8, 16, 64, 128] on both SST-2 and IMDB datasets.

#### rSVD Training
```bash
cd rSVD
python rsvd_call.py
```

This will train rSVD models with ranks [4, 8, 16, 64, 128] on the SST dataset.

#### rSVD + SVT (Fixed Rank)
```bash
cd rSVD_SVT_Fixed_Rank
python rsvd_svt_call.py
```

#### rSVD + SVT (Rank Fractions)
```bash
cd rSVD_SVT_Rank_Fractions
python rsvd_svt_call.py
```

#### rSVD with Weight Reduction
```bash
cd rSVD_Weight_Reduction
python rSVD_call.py
```

#### rSVD with Weight Reduction + SVT
```bash
cd rSVD_Weight_Reduction_SVT
python rSVD_call.py
```

### C. Evaluation

Evaluation is performed automatically during training. Results are saved in the `graph/` directory for each approach:

- **Loss curves**: `graph/r{rank}/epoch_loss.csv`
- **Memory usage**: `graph/r{rank}/epoch_peak_memory.csv`
- **FLOPs statistics**: `graph/r{rank}/flops_profiler_stats.csv`
- **Forward pass metrics**: `graph/r{rank}/forward_pass/`

To visualize results, use the plotting scripts in each `graph/` directory:

```bash
cd Lora/graph
python plot.py

cd rSVD/graph
python multi-rank-plot.py
python plot_memory_breakdown.py
python plot_optimizer_memory.py
```

---

### D. Quickstart: Minimum Reproducible Result

To reproduce the **rSVD rank-16 result on SST-2** (our best memory-accuracy trade-off):

```bash
# Step 1: Set up environment
cd rSVD
pip install -r requirements.txt

# Step 2: Datasets are downloaded automatically by Hugging Face
# No manual download needed

# Step 3: Run training
python rsvd_call.py

# Step 4: View results
cd graph/r16
cat epoch_loss.csv
cat epoch_peak_memory.csv
```

**Expected Results**:
- Final training loss: ~0.0727
- Final evaluation loss: ~0.3252
- Peak memory: ~1.85 GB
- Optimizer memory savings: ~98% vs. standard Adam

---

## 5. Notes

### Project Structure
```
HPMLProj/
├── Lora/                    # LoRA implementation
│   ├── lora_config/        # LoRA configuration
│   ├── forward_pass/       # Forward pass profiling
│   └── graph/              # Results and plots (sst2, imdb subdirs)
├── Lora+TopR/              # LoRA + TopR combination
│   └── graph/              # Results and plots
├── rSVD/                   # Core rSVD implementation
│   ├── rsvd_config/        # rSVD configuration
│   ├── optimizer/          # rSVD optimizer
│   ├── forward_pass/       # Forward pass profiling
│   └── graph/              # Results, plots, and metrics
├── rSVD_SVT_Fixed_Rank/    # rSVD with fixed-rank SVT
│   ├── rsvd_svt_config/    # SVT configuration
│   ├── sst/graph/          # SST-2 results
│   └── imdbb/graph/        # IMDB results
├── rSVD_SVT_Rank_Fractions/ # rSVD with adaptive rank SVT
│   ├── rsvd_svt_config/    # SVT configuration
│   ├── sst/graph/          # SST-2 results
│   └── imdbb/graph/        # IMDB results
├── rSVD_Weight_Reduction/   # rSVD with weight compression
│   ├── rSVD_HF_config/     # Hugging Face config
│   └── graph/              # Results and plots
├── rSVD_Weight_Reduction_SVT/ # Combined approach
│   ├── rSVD_HF_config/     # Hugging Face config
│   └── graph/              # Results and plots
├── custom_adam/            # Custom optimizer implementations
└── docker/                 # Docker configuration
```

### Trained Models
- Trained models are saved in each approach's directory with naming pattern:
  - `distilbert-{dataset}-{method}-r{rank}/`
- Example: `rSVD_Weight_Reduction/distilbert-sst2-rSVD-r16/`

### Memory Profiling
- Memory breakdown analysis available in `rSVD/graph/memory_breakdown_r{rank}.csv`
- Optimizer memory comparison in `rSVD/graph/optimizer_memory_comparison.csv`


## Acknowledgments

This project was completed as part of the High Performance Machine Learning course. We thank the course staff for their guidance and support.

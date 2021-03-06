# M2A: Motion Aware Attention for Video Action Recognition [[ArXiv]](https://arxiv.org/abs/2111.09976)

Code is built on top of the TSM module's repo (https://github.com/mit-han-lab/temporal-shift-module). 

We use the following code for I3D experiments: https://github.com/IBM/action-recognition-pytorch.

Key Files: 
- `main.py`: Train using DataParralel
- `opts.py`: command line arguments for training 
- `ops/sota.py`: SOTA temporal modules which we compare against 
- `ops/temporal_shift.py`: M2A, other temporal modules, and inserting the temporal modules into the network 
  - `CustomMotionAttention` is M2A
- `scripts/`: training script examples 



# FBS Implementation

Dynamic Channel Pruning: Feature Boosting and Suppression (ICLR'19) Implementation 

## environment
  - python3.6
  - pytorch 
  - torchvision 
  - tqdm
  - numpy 

I implement the training code. However output-sparsity during the inference time and MACs calculation isn't implemented in this code. Results appended in this ReadMe code are calculated using code [2].

You can execute code with following instruction. 
```code
python main.py --fbs [fbs] --sparsity_ratio [r] --lasso_lambda [lambda] --epochs [E] --batch_size [B] --lr [LR] --seed [Seed] --num_worker [N] --ckpt_path [path]
```

[1] Dynamic Channel Pruning: Feature Boosting and Suppression (https://openreview.net/pdf?id=BJxh2j0qYm)     
[2] FBS pytorch code(https://github.com/sehkmg/FBS-torch)

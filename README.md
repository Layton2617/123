# Lab 2 – High Performance Machine Learning

This repository contains all the code and instructions for COMS 6998 Lab 2, which focuses on profiling and optimizing machine learning workloads using PyTorch. The experiments (C2–C8) and theoretical questions (Q1–Q5) are designed to be executed in a consistent computing environment (with or without GPU).


## How to Run Experiments

All experiments are designed to be run from the command line. Ensure all code files are in the same directory and that you run them in the same computing environment for consistency. For GPU-based experiments, verify that your machine has a GPU available.

### **Run All Experiments Sequentially**
```bash
python3 C2.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 2 --optimizer sgd --epochs 5 \
&& python3 C3.py --data_path ./data --batch_size 128 --workers "0,4,8,12,16" \
&& python3 C4.py --use_cuda True --data_path ./data --batch_size 128 --workers "1,8" \
&& python3 C5.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 2 --optimizer sgd --epochs 5 --device gpu \
&& python3 C6.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 8 --epochs 5 \
&& python3 C7.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 8 --epochs 5 \
&& python3 C8.py --use_cuda True --data_path ./data --batch_size 128 --workers 8 --epochs 10
```

This command will execute all experiments (C2-C8) sequentially in a single run.

### **Running Individual Experiments**
If you want to run a specific experiment separately, you can use the following commands:

#### **C2: Training with Timing Measurements**
```bash
python3 C2.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 2 --optimizer sgd --epochs 5
```

#### **C3: I/O Optimization**
```bash
python3 C3.py --data_path ./data --batch_size 128 --workers "0,4,8,12,16"
```

#### **C4: Profiling – Data-loading vs. Training Time**
```bash
python3 C4.py --use_cuda True --data_path ./data --batch_size 128 --workers "1,8"
```

#### **C5: GPU vs. CPU Training**
Run on GPU:
```bash
python3 C5.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 2 --optimizer sgd --epochs 5 --device gpu
```
Run on CPU:
```bash
python3 C5.py --use_cuda False --data_path ./data --batch_size 128 --num_workers 2 --optimizer sgd --epochs 5 --device cpu
```

#### **C6: Experimenting with Different Optimizers**
```bash
python3 C6.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 8 --epochs 5
```

#### **C7: Experimenting Without Batch Norm**
```bash
python3 C7.py --use_cuda True --data_path ./data --batch_size 128 --num_workers 8 --epochs 5
```

#### **C8: Accelerate with torch.compile**
```bash
python3 C8.py --use_cuda True --data_path ./data --batch_size 128 --workers 8 --epochs 10
```

To view the profiling trace (if enabled), run:
```bash
tensorboard --logdir=./log/profiler
```



 


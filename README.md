# Replay Memory Regulation for Action Advising

## Requirements
**Dependencies**
- numpy
- opencv-python
- tensorflow=1.13.1  

**Environments**  
- [LavaWorld](https://github.com/ercumentilhan/LavaWorld) 
- [MinAtar](https://github.com/ercumentilhan/MinAtar) 

## Execution

A training process can be started as follows with appropriate argument(s):
```
python main.py --experiment-setup <experiment setup>
```

The list of all hyperparameters can be found in `main.py`.

**Format of _experiment-setup_:**  
This an integer argument with three digits **abc** (except for _no advising_ which takes 0) which defines the setup of the experiment in terms of action advising method, replay memory regulation method, and action advising budget to be used.

- **a:** Action advising method
  - **1:** Early advising
  - **2:** Uniformly random advising
- **b:** Replay memory regulation method
  - **0:** None
  - **1:** Counter
  - **2:** RND
- **c:** Budget (these are defined in `executor.py`)
  - **0:** 500
  - **1:** 1000
  - **2:** 2500
  - **3:** 5000
  - **4:** 10000
  - **5:** 25000
  - **6:** 50000
  - **7:** 100000

**Experiment setups used in the study:** 
- **LavaWorld**
  - **No Advising:** 0
  - **Early Advising:** 101, 104, 105, 111, 114, 115, 121, 124, 125
  - **Uniformly Random:** 201, 204, 205, 211, 214, 215, 221, 224, 225
  
- **MinAtar (for each game)**
  - **No Advising:** 0
  - **Early Advising:** 104, 105, 107, 124, 125, 127
  - **Uniformly Random:** 204, 205, 207, 224, 225, 227

# snaker
Automatic snake game using reinforcement learning
## Training
This model has been trained until the graph flatlined
## Evaluation
#### To run the model
### 1. Edit `model_eval.py`
In line 8, find

    model = torch.jit.load('model.pt')
replace `'model.pt'` with the complete path

### 2. Run

    python model_eval.py

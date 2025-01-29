# Run guide for the project

This guide provides detailed instructions on how to run the `train_and_test.py` script with various command-line arguments to customize the training and testing process.

## Step 1: Set up your environment

Ensure that you have all the dependencies installed. It is recommended to create a virtual environment and install the necessary libraries (e.g., `torch`, `yaml`, etc.).

### Create and activate a virtual environment (optional)
- python -m venv venv
- source venv/bin/activate
  
### On Windows: 
- venv\Scripts\activate

### Install the required dependencies
- pip install -r requirements.txt

## Step 2: Configuration file

Before running the script, you can have a look at configuration YAML file (`config/Distributed_Lenet.yaml`). This file contains the settings for your model and experiment. 

## Step 3: Running the script

You can execute the script in the `src` folder using the following command:

- python train_and_test.py

### Available Arguments

- `--cpu`: Use CPU as the device for training and testing.  
- `-lr, --learning-rate`: Override the default learning rate.  
- `-bs, --batch-size`: Override the default batch size.  
- `-ep, --epochs`: Override the default number of epochs for training (which is 150).  
- `-opt, --optimizer`: Override the default optimizer: you can choose among AdamW, SGDM, LARS, LAMB. Local methods will not use this argument, as LocalSGD is required.
- `-sch, --scheduler`: Override the default learning rate scheduler (we used CosineAnnealingLR, but we implemented PolynomialDecayLR as well).  
- `-p, --patience`: Override the early stopping patience: we generally set it to 0, but it can be useful sometimes in order to prevent overfitting.  
- `-wu, --warmup`: Override the number of warm-up steps for the learning rate scheduler: this is useful for large batch optimizers. 
- `-wd, --weight-decay`: Override the default weight decay.  
- `-sm, --slowmo-momentum`: Override the momentum (beta) for the SlowMo optimizer: if nothing is specified, we use 0.0
- `-slr, --slowmo-learning-rate`: Override the learning rate (alpha) for the SlowMo optimizer: if nothing is specified, we use 1.0
- `-nw, --number-workers`: Override the number of workers: use 0 for pure centralized, 1 for a distributed setting equivalent to centralized
- `-wss, --worker-sync-step`: Override the number of worker sync steps: useful only in distributed setting
- `-wls, --worker-local-step`: Override the number of local steps per worker: useful only in distributed setting  
- `-wbs, --worker-batch-size`: Override the batch size for workers: we use 64
- `-dls, --dynamic-local-step`: Override the dynamic local step function for workers: you can choose among those listed at the end of `selectors.py`, in `src/utils`  
- `-P, --pretrained`: Path to the pretrained model checkpoint.  
- `-LC, --load-checkpoint`: Resume training from the last checkpoint.  
- `-T, --test`: Skip training and run only the testing phase.  
- `-en, --experiment-name`: Change the name of the experiment.  
- `-v, --version`: Update the experiment version.
- 
## Step 4: Example Full Command

Here is an example of a full command that customizes several arguments:

```bash
python train_and_test.py -lr 0.002 -bs 256 -ep 150 -opt LAMB -sch CosineAnnealingLR -nw 0
```

This command:
- Sets the learning rate to `0.002`
- Sets the batch size to `256`
- Trains for `150` epochs
- Uses the `LAMB` optimizer and `CosineAnnealingLR` scheduler
- Uses the centralized training function

## Step 5: Results

After running the script, you should see logs that indicate the training progress (if you are not using the `-T` flag). Otherwise, it will run the testing phase and output the results. 
The experiment results will be saved in the directory specified in the configuration file.
Finally, a plot will be saved in the corresponding directory.

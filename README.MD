# Few-shot learning of Sketch based Face image matching using Distance-guided visual embedding

## 1. Data Pre-processsing
    bash preprocess.sh
Data will be automatically pre-processed and be stored in Data/Preprocessed/.
## 2. Training
    cd Code
    python main.py
+ There are many parameters below. (The default setting is what we got the best result.)
    + Random seed (e.g. --seed 1)
    + Epoch (e.g. --epochs 50)
    + Batch size (e.g. --batch_size 100)
    + Learning Rate (e.g. --lr 0.001)
    + Guide (e.g. --guide None or --guide Distance)
    + Neural Network (e.g. --neural_net ResNet-152)
        + One of (VGG-11, VGG-13, VGG-16, VGG-19, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152)
    + Device (e.g. --device cpu or --device cuda:1)
    + You can check more parameters on Code/main.py line14:30.
+ You can apply parameters by adding arguments. For example,
```
python main.py --lr 0.001 --neural_net ResNet-50 --seed 1
```
+ Right after the training is started, new directory will be created on 'Results' folder. the name of the new(most recent) directory is 'Result{largest number}'. Inside the directory, followings will be created during training.
    + hparam.json: Hyperparameters.
    + TrainDistanceDensity: Stores distance density of the training set on each epoch.
        + Since training period contains batch normalization or dropout, the distance density is different from density on evaluation period.
    + Log.jpg: Contains Loss, Average distance and Accuracy on each epoch.
    + model.pth: Weights of the neural network when it has the best few shot accuracy.
    + model0.pth: Weights of the neural network when it has the best zero shot accuracy.
    + Training_Log.txt: Contians Time taken, Loss, Statistics of Distance distribution and Accuracy on each epoch.
    + history.pkl: Contains the values used to write Training_Log.txt as a pickle file in order to load easily.
## 3. Evaluation
    python reviewer.py --result 1 --max_thres 10
> The argument 'result' means evaluation of "Results/Result{result}". (**Check the Directory**), After evaluation, increase or decrease 'max_thres' and then re-evaluate if the range of x axis of Distance Density on review.jpg is insufficient or too much.
+ Basic Metrics such as Accuracy or Average Distance are calculated during training. However, we sometimes need to add or fix some metrics even after training is done. so metrics taking a lot computation such as MR, MRR, Hit@K, Distribution of Distances are separated.
+ The following files will be created after evaluation is done.
    + review.jpg: Contains MR, MRR, Accuracy, Hit@K, Distance Density using model.pth which has shown the best accuracy on few shot setting.
    + review.pkl: Contains the values used to plot graphs on review.jpg in pickle file in order to load easily.
    + review0.jpg: Contains MR, MRR, Accuracy, Hit@K, Distance Density using model.pth which has shown the best accuracy on zero shot setting.
    + review0.pkl: Contains the values used to plot graphs on review0.jpg in pickle file in order to load easily.

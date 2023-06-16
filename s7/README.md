
## Genral Good Model Build Optimsation Approch 
Set up -> Basic Skeleton -> 
Some quick tips

| Stage        | Target           | Results  | Analysis  | 
| ------------- |:-------------:| -----:|-----:|
| Set up   | <ul><li>Get the set-up right</li><li>Set Transforms</li><li>Set Data Loader</li><li>Set Basic Working Code</li><li>Set Basic Training  & Test Loop</li></ul> | <ul><li>Parameters: 6.3M</li><li>Best Training Accuracy: 99.99</li><li>Best Test Accuracy: 99.24</li></ul>|<ul><li>Extremely Heavy Model for such a problem</li><li>Model is over-fitting, but we are changing our model in the next step</li></ul> | 
| Basic Skeleton  |<ul><li>Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible. </li><li>No fancy stuff</li></ul>|<ul><li>Parameters: 194k</li><li>Best Train Accuracy: 99.35</li><li>Best Test Accuracy: 99.02</li>|<ul><li>The model is still large, but working. </li><li>We see some over-fitting</li>
| Lighter Model  |<ul><li>Make the model lighter</li></ul></ul>|<li>Parameters: 10.7k</li><li>Best Train Accuracy: 99.00</li><li>Best Test Accuracy: 98.98</li></ul>|<li>Good model!</li><li>No over-fitting, model is capable if pushed further</li></ul> 
| Batch Normalization  |<ul><li>Add Batch-norm to increase model efficiency.</li></ul>|<ul><li>Parameters: 10.9k</li><li>Best Train Accuracy: 99.9</li><li>Best Test Accuracy: 99.3</li>|<ul><li>We have started to see over-fitting now. </li><li>Even if the model is pushed further, it won't be able to get to 99.4</li></ul>
| Regularization  |<ul><li>Add Regularization, Dropout</li></ul>|<ul><li>Parameters: 10.9k</li><li>Best Train Accuracy: 99.39 (20th Epoch) & 99.47 (25th)</li><li>Best Train Accuracy: 99.30</li></ul>|<ul><li>Regularization working. </li><li>But with the current capacity, not possible to push it further. </li><li>We are also not using GAP, but depending on a BIG-sized kernel</li></ul>
| Global Average Pooling  |<ul><li>Add GAP and remove the last BIG kernel.</li></ul>|<ul><li>Parameters: 6k</li><li>Best Train Accuracy: 99.86</li><li>Best Test Accuracy: 98.13</li></ul>|<ul><li><span style="color:red">Adding Global Average Pooling reduces accuracy - WRONG</span></li><li>We are comparing a 10.9k model with 6k model. Since we have reduced model capacity, a reduction in performance is expected. </li></ul>
| Increasing Capacity  |  | | | 
| Correct MaxPooling Location  |  | | | 
| Image Augmentation  |  | | | 
| Playing naively with Learning Rates  |  | | | 

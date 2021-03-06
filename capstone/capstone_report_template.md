# Machine Learning Engineer Nanodegree
## Telstra Network Disruptions
Jeremias Binder 
June 2nd, 2017

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

Telstra is australias biggest telecommunications and media company with an annual revenue of over 20 billion USD (A$27.1 billion).
In a recent Kaggle competition, Telstra challenged people to predict the severity of service disruptions on their network. The reason:

Telstra is on a journey to enhance the customer experience - ensuring everyone in the company is putting customers first. In terms of its expansive network, this means continuously advancing how it predicts the scope and timing of service disruptions. Telstra wants to see how kaggle users would help it drive customer advocacy by developing a more advanced predictive model for service disruptions and to help it better serve its customers.

<p>(Telstra Network Disruptions. (2016). Retrieved and edited June 12, 2017,  from the <a href="https://www.kaggle.com/c/telstra-recruiting-network/data">competitions page</a>).</p>

The data used in the competition is provided by Telstra. Service logs and data from their network nodes were prepared, anonymised and are available on <a href="https://www.kaggle.com/c/telstra-recruiting-network/data">Kaggle</a>.

The competition ended at 2/29/2016, the winner was a user called ‘Mario Filho’. In total, 974 teams participated in the competition. 


One of the reasons i choose this project as my capstone project for the Machine Learning Nanodegree at Udacity, is that the company i am currently working at, the sovanta AG in Heidelberg, Germany, has some projects running in the domain of predictive maintenance.



----------------
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:

- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

----------------


### Problem Statement

The problem we are about to solve, is to minimize the reaction time to interruptions in Telstras network. Long reaction times lower customer satisfaction in the long run and can therefore be costly to Telstra. 
A model, that can predict accurately the network failures in advance, would be very valuable to Telstra and its customers.

The problem right now is, that Telstra has no good estimation on when one of their nodes will fail. They might have clues (certain nodes reporting an error), but no further conclusions are drawn from this data.

A predictive model can change that: The data provided by the nodes will be used to create a model, that will accurately predict failures on the network.

Since the input information is digitally obtained, each error message can be put in a certain category and is distinct. Since it’s a future prediction, it’s easily verifiable: A er the event is predicted, the actual time and place can be observed and the degree to which the prediction is correct can be verified.

----------------
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

----------------

### Metrics

The thing Telstra is interested in, is when and where their nodes are likely to fail. The quality of the prediction therefore matters: 
To determine the quality of such a model the multi-class logarithmic loss is used. Each data row has been labeled with one true class, which represents the severity of the incident (an incident with label '0' means, there is no issue). For each row,  a set of predicted probabilities is submitted (one for every fault severity). The formula is then,

\\[ logloss = - {1\over N} \\sum_{i=1}^{N} {\\sum_{j=1}^{M}}  {y_{ij}  log (p_{ij})},\\]


where N is the number of rows in the test set, M is the number of fault severity classes,  $ log $ is the natural logarithm, $ y_{ij} $ is 1 if observation $i$ belongs to class $j$  and $0$ otherwise, and $p_{ij}$ is the predicted probability that observation $i$ belongs to class $j$.
Or in layman terms: The lower the sum of wrongly predicted severities, the lower the logloss. Participants with a lower logloss on the testset get ranked higher on the public leaderboard. 
Scoring a low logloss on the testset is important, since the model might overfit on the trainset.

Why does the multilogloss makes sense here?
The output of our model is the severity of a network issue ("Fault_Severity"). This severity is measured in three integer categories from zero to two. Meaning, we have exactly three labels. 
But: Our model will give us predictions in the form of a probability distribution (i.e. 0: 0.212, 1: 0.573, 2:0.215). The most likely category in this example would be the second one with a chance of 0.573. If we were to measure the output label (in this example "1"), some valuable information about the quality of our model is discarded. 

Now, intuitively speaking:  log-loss is the cross entropy between the distribution of the true labels and the predictions. We use it to gauge the “extra noise” that comes from using a predictor as opposed to the true labels. By minimizing the cross entropy, one maximizes the accuracy of the classifier.



----------------
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

----------------


## II. Analysis
_(approx. 2-4 pages)_


### Data Exploration
Lets take first glance at the data, which comes in six files:

train.csv - the training set for fault severity 
test.csv - the test set for fault severity 
event_type.csv - event type related to the main dataset 
log_feature.csv - features extracted from log files 
resource_type.csv - type of resource related to the main dataset 
severity_type.csv - severity type of a warning message coming from the log

All the files have one attribute, that connects them all together: The column named 'id'. Note that the 'id' doesn't have the same property as a key we might now from relational databases: Sometimes one id is listed several times in one column.

After quickly scrolling through the data in MS Excel, i decided to do the following:
In order to just process numeric data, i dropped the strings from all the csv files containing the description ('event_type ', 'log_feature ', 'resource_type') manually with the replace function.

| id   |     event_type      |  
|----------|:-------------:|
| 62093 |  ~~event_type~~ 15 | 

The thing i did then was calculating basic statistics of the data from the files to get a good first understanding.
What i found out was:

 - Most of the interruptions in the *train* data are just a temporary network glitch ( Fault
   Severity = 0), but 1 and 2 do seem to cause problems quite frequently
   (around 35% of the time). 
   
 - The *event types* 11,35 and 34, 15 and 2 seem
   occur a lot together with a network incident
 - *Resource types* 8 and 2 are the most prevalent. There is a significant gap
   between these two resources and the rest.

 - *Severity type* of warning messages 1 and 2 seem to be the most frequent ones.
 - *Log features* is actually tricky to deal with: we have the 'id' column, then the 'log feature' column and a 'volume' column, which indicates how often a 'log feature' appeared with an network incident. The fact that there are 386 different 'log features' became a real sticking point for this problem. 

Note that this is just a univariant analysis of the features provided in the single files. The initial plan was to provide a multivariant analyis of the data, but this became almost impossible, since the data is quite sparse.

So the challenge in the first place was to merge the files into one coherent frame to work with. This turned out to be more difficult than expected, since the the *log_feature* table contained 453 categorical values, which couldn't be merged easily together with the ids of the other files.

Also, each id usually had just a few features from *log_feature*, so i decided to one-hot encode the information into the frame. Actually, i tried two versions of one-hot-encoding to solve this problem:
At first, i did a binary one-hot-encoding and discarded the *volume* information of the features, then i tried something else and used the *volume* value of each feature instead of just a one. 

I ended up with a dataframe with a total of 453 features. Of course, this is just the first draft of the set, so started to think of ways to further reduce the feature space.

----------------
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_
- ----------------

### Exploratory Visualization

As already explained above, due to the sparse nature of the data, a multivariate analysis of the data is not easy to provide. However, i plotted the features in histograms, which seems like a good entry point for further discussions. 

![enter image description here](https://image.ibb.co/df4Jk5/fig.png)
##TODO : Log feature top 50

--------------

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

----------
### Algorithms and Techniques

So, after merging the files in one single frame, i could finally get started with my very first model. Reading through some of the forums on the web and in specific kaggle, i decided to take a closer look  on the xgboost algorithm.

The xgboost algorithm is a flexible, performant, and scalable gradient boosting algorithm, which gained a lot of popularity on recent kaggle competitions. Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made.

Lets have a look on the most important default parameters set in our xgboost classifier ( or read the [full list](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)):

 - eta [default=0.3, alias: learning_rate]: The learning rate of algorithm, Makes the model more robust by shrinking the weights on each step.
 - gamma [default=0, alias: min_split_loss]: A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
 - max_depth [default=6]: This describes the maximum depth of a decision tree. The parameter is used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample. **Include in CV**
 - min_child_weight [default=1]: here we define the minimum sum of weights of all observations required in a child node of a DT. Its also a measure to prevent overfitting, it avoids relations which might be highly specific to a certain sample.  **Include in CV**
 
What the xgboost algorithm does, with 


----------
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

 ----------

### Benchmark

To determine the quality of our model the [multiclass logarithmic loss function](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) is used. Each data row has been labeled with one true class, which represents the severity of the incident (an incident with label '0' means, there is no issue). For each row,  a set of predicted probabilities is submitted (one for every fault severity). The formula is then,

\\[ logloss = - {1\over N} \\sum_{i=1}^{N} {\\sum_{j=1}^{M}}  {y_{ij}  log (p_{ij})},\\]


where N is the number of rows in the test set, M is the number of fault severity classes,  $ log $ is the natural logarithm, $ y_{ij} $ is 1 if observation $i$ belongs to class $j$  and $0$ otherwise, and $p_{ij}$ is the predicted probability that observation $i$ belongs to class $j$.
Or in layman terms: The lower the sum of wrongly predicted severities, the lower the logloss. Participants with a lower logloss on the testset get ranked higher on the public leaderboard. 
Scoring a low logloss on the testset is important, since the model might overfit on the trainset. 

Once the model is fitted, we will run the built-in xgboost function 'predict_proba' to predict the probabilities for each severity category in the 'test.csv' file and then upload them on the kaggle submission page, to calculate against the kaggle testset.

To get a feeling on what score to expect: 
The first place on the Telstra leaderboard, a user called Mario Filho, scored a log loss of 0.3954.
Once i submitted the complete dataset & ran xgboost with default settings, i scored a logloss of 0.58323. So the final prediction will probably be in the range between 0.4 and 0.55.

 ----------
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_
 ----------

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

####1. Merging the data.

As explained in the 'Data Exploration' Section, the problem with our dataset is, that the length of the different data files attributes have different length. Not every id in the one table corresponds to every id in the other tables. Some tables vary in length. In order to fit the data to the xgboost classifier, the dataframe first has to be put in shape.

To unify this information, i decided to one-hot encode all the features. While this was quite straightfoward with *events*, *severity*, and *resource type*, i had decided to change plans with *log features* at first:
I decided to drop the information of the *volume* column and make *log feature* binary.

Say for a given *id* the volume of feature 2 is 7, while log features 1-n are all zero. You can see the pure "one-hot encoded" approach in the first row, and the "volume-hot-encoded" approach in the second row.
<br>

| approach   || id   |     log feature 1     |       log feature 2    |      ...    |        log feature n    | 
|----------||----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| one-hot encode | | 62093 |  0 |  1 | ... | 0 |
| volume-hot encode | | 62093 |  0 |  7 | ... |0 |

<br>

Now i had all the features one hot encoded in different tables.
Time, to unify them into one table. In order to do so, i merged the files one by one into the 'train' pandas frame. Here's a code snippet:

    train = pd.merge(train, pd.DataFrame(severity, columns = ['id', 'severity_type']), on='id',how='left')

 As you can see, i am merging the 'train' frame with the 'severity' frame on their id, by using keys from the train frame, similar to a SQL left outer join. The operation preserves the key order.

I finally ended up with a 7381x453 dataframe, which contained all possible features, one-hot encoded.
So here's a simplified version of how the table looks after all the operations. Note that it containts 453 columns (shortened here with 1...n) 

| id|| severity <br>type 1 ... n   |event <br>type 1 ... n |log <br>feature 1 ... n|  resource <br> type 1 ... n|fault <br>severity| 
|-||-|:--:|:--:|:--:|:----:|:-----:|
| 62093| | 1 |  0|  1 | 1 | 0 |
| 58261 | | 1 |  0 |  0 | ... |2 |
| ... | |  ... |  ... |   ... | ... | ...|
| 28413 | |  0|  1 |   0 | 0 | 1|

This dataset is now ready for a first fitting iteration with xgboost.

 ----------
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_
 ----------

### Implementation

 Now that i finally got the dataframe set up, we're ready to run the first iteration of our fault severity prediction.
 As described above, i choose the xgboost algorithm as classifier. Lets split our dataset into a train- and testset, and see what logloss we can achieve with this rough first draw:

    test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=test_size, random_state=7)
	model_1 = xgboost.XGBClassifier(eval_metric='mlogloss', max_depth = 5)
	model_1.fit(X_train, y_train.values.ravel())

Note that the initial parameters for our classifier are the default settings, i just changed *max_depth* to *5*, since the default value of *3* appeared very small for such a sparse dataset. Parameter tuning and model improvement will be the topic of the refinement chapter.

The moment of truth came, the classifier accepted the frame without an error and fitted the first classifier. 

	   from sklearn.metrics import log_loss
	   y_pred = model_1.predict_proba(X_test)
	   print("Logloss: %.3f" % log_loss(y_test,y_pred))
	   ----
	   Logloss: 0.584

A value of 0.584 on the personal testset and 0.578 on the kaggle board is a quite acceptable value for the first submission without any tuning. 
 ----------
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_
 ----------
### Refinement

The initial model is now set up, and gives us a stable result we can further build upon. 
As in the preprocessing chapter described, there are a lot of attributes we included, resulting in a very sparse dataset. In order to improve the result, it is obvious to me, that the feature space needs to be reduced.

 ----------
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_
 ----------

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

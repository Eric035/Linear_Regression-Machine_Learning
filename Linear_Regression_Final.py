# By: Cheuk Hang Leung (Eric), Donovan Chiazzese, Ingrid Feng

import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import timeit
import pickle as pk
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as pp
np.set_printoptions(threshold=100)

with open("proj1_data.json") as fp:
    data = json.load(fp)

# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment

# Example:
# data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
# for info_name, info_value in data_point.items():
#    print(info_name + " : " + str(info_value))

# ----------------------------------------------------------------------------#
# Splitting the first 10,000 points for training set.
training_set = data[0:10000]  # training set is a list of dictionary, which contains the partition (index 0 - 9999) datapoints of our given data

# Splitting the next 1,000 data points for validation set
validation_set = data[10000:11000]  # validation set is a list of dictionary, which contains the partition (index 10000 - 10999) datapoints of our given data

# Splitting the last 1,000 for test set
test_set = data[11000:12000]            # test set is a list of dictionary, which contains the partition (index 11000 - 11999) datapoints of our given data

# ---------------------------------------------------------------------------#
# Encoding The features children, is root, and controversality
# To encode our training set
for index in range(10000):
    if training_set[index].get("is_root") == True:  # If a comment is a root, we set the is_root variable to 1.
        training_set[index]["is_root"] = 1
    else:
        training_set[index]["is_root"] = 0

# Encode validation set
for index in range(1000):
    if validation_set[index].get("is_root") == True:
        validation_set[index]["is_root"] = 1
    else:
        validation_set[index]["is_root"] = 0

# Encode test set
for index in range(1000):
    if test_set[index].get("is_root") == True:
        test_set[index]["is_root"] = 1
    else:
        test_set[index]["is_root"] = 0

# ---------------------------------------------------------------------------#
# Pre-process the text message

def lowerCaseAndSplit(text):                # A simple function that takes in a text then lower-casing and splitting the text on whitespcae token. It returns a list of strings.
    wordList = (text.lower()).split()       # Storing our result into words
    return wordList

def splitIntoComponents(text):              #A function that takes in a comment and split into components separated by comma, period, etc. not including whitespace
    while 1:
        try:
            last = text[-1]
        except:
            break
        else:
            if type(last) is not int and type(last) is not float and not last.isalpha():
                text = text[:-1]
            else:
                break
    text = text.replace(";", ",").replace(".", ",").replace("!", ",").replace("?", ",").replace(":", ",")
    components = text.split(",")
    return components

def splitNoRepitition(text):
    text = text.replace(";", "").replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace(":", "").replace("(", "").replace(")", "")
    regular = lowerCaseAndSplit(text)
    unique = set(regular)
    return regular, unique

# -----------------------------------------------------------------------------#
# Word Count per comment
def countWordPerComment (dataSet):                  # A function that takes in a data set and count how many words in each comment then output the result dictionary
    numWordsPerComment = {}                         # A Dictionary to store number of words for each comment
    for example in range(len(dataSet)):
        wordsInComment = len(splitNoRepitition(dataSet[example].get("text"))[0])
        numWordsPerComment["example " + str(example)] = wordsInComment
    #print(numWordsPerComment)
    return numWordsPerComment

dictionaryWords = countWordPerComment(test_set)     # (A dictionary of words) Input to test our frequencyWordCount method

                                                    # A method that takes in a word dictionary and count how many comments have n word (n: 1,2,..., maximumWordsInDataSet) and output a dictionary
def frequencyWordCount (dictionaryWords):
    frequencyDict = {}
    wordListTuple = sorted(dictionaryOfWords.items(), key=lambda x: x[1], reverse=False)    # Sort our dictionary by value and store it into a list of tuples in ascending order
    maxNumWords = wordListTuple[len(wordListTuple)-1][1]                                    # We need the maximum number of words among the comments in the input data set for our loop
    
    for i in range(maxNumWords+1):                                                          # Since range (1, maxNumWords (exclusive)), we need to add 1
        frequencyDict["Number of comments with " + str(i) + " words: "] = 0
    numWords = 0
    while  numWords <= maxNumWords:                                                         # While loop to check whether our comment has the same word as our index
        for i in range(len(wordListTuple)):                                                 # For loop for looping through each comment
            if (wordListTuple[i][1] == numWords):
                frequencyDict["Number of comments with " + str(numWords) + " words: "] += 1
        numWords += 1
    return frequencyDict
# ---------------------------------------------------------------------------#

def genMatrix(dataSet):                                                     # A generic function that takes in a list of dictionaries then outputs a matrix X (each row is a sample and each column represents a feature) and a vector Y (target values)
    wordFreqDict = {}                                                       # A dictionary to store the frequency of every word from all comments
    for example in range(len(dataSet)):                                     # Loop through each comment
        for w in lowerCaseAndSplit(dataSet[example].get("text")):           # Loop through each word of each comment
            if w in wordFreqDict:                                           # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
                wordFreqDict[w] += 1
            else:                                                           # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
                wordFreqDict[w] = 1

    wordListTuple = sorted(wordFreqDict.items(), key=lambda x: x[1],reverse=True)   # Sort our dictionary by value and store it into a list of tuples
    wordListTuple = wordListTuple[0:160]                                            # We only need the top 160 values(frequencies) from our list

    numRow = (len(dataSet))                                                         # Number of rows of the output matrix X depends on how many examples we have.
    numCol = (len(dataSet[0]) + len(wordListTuple) - 1 + 4)                         # Number of columns depends on number of features (164): is_root, controversiality, children, bias, w0_freq, w1_freq,...,w160_freq
    matrix_X = np.zeros((numRow, numCol))                                           # A matrix with all zeros as its entries.
    vector_Y = np.zeros((numRow, 1))                                                # Initialize a vector Y with size numRow
    dictionaryOfWords = countWordPerComment(dataSet)
    for example in range(numRow):                                           # Loop through each example in the data set
        matrix_X[example][0] = dataSet[example].get("is_root")              # First index of each row (example) is the binary is_root
        matrix_X[example][1] = dataSet[example].get("controversiality")     # Second index of each row is the controversiality
        matrix_X[example][2] = dataSet[example].get("children")             # Third index of each row is the number of children (replies) the comment received
        matrix_X[example][3] = 1  # Bias term

        # Our new features
        
        # New features 1 - Controversiality squared
        controversiality_squared = matrix_X[example][1]
        matrix_X[example][4] = np.square(controversiality_squared) + 0.001

        # New features 2 - Children squared         (max training = 43,    max validation = 30)
        children_squared = matrix_X[example][2]
        matrix_X[example][5] = np.square(children_squared)/30

        # New feature 3 - Number of words per comment inverse
        matrix_X[example][6] = 1 /float((dictionaryOfWords["example " + str(example)] + 1))

        # New feature 4 - number of repeated words  (max training = 587,   max validation = 249
        curr_text = dataSet[example].get("text")
        comment = lowerCaseAndSplit(curr_text)
        wordCountList = np.zeros((1, numCol))  # Temporary vector to store number of time each frequent word appears in that particular comment


        regular, unique = splitNoRepitition(curr_text)
        matrix_X[example][7] = (len(regular) - len(unique))/243

        vector_Y[example] = dataSet[example].get("popularity_score")  # Target variable: Popularity score of the comment

        col = 8 # Variable to keep track of which column(feature) we are at
        word = 0  # A string to loop through our wordListTuple for our counting
        while (word < len(wordListTuple)) and (col < numCol):
            matrix_X[example][col] += float(comment.count(wordListTuple[word][0]))/float(wordListTuple[word][1])

            col += 1
            word += 1

    return matrix_X, vector_Y

# PART 2
currentSet = test_set

if currentSet == test_set:
    print("Test set")
if currentSet == training_set:
    print("Training set")
if currentSet == validation_set:
    print("Validation set")

XY = (genMatrix(currentSet))                                                # Storing the return values of genMatrix into XY variable. The return value is a tuple.


Y = np.array(XY[1])                                                         # Y vector returned by genMatrix func, will be referred to as Y
X = np.array(XY[0])                                                         # X matrix returned by genMatrix func, will be referred to as X

num_of_examples = X.shape[0]                                                # The size of our Y vector is how many examples we have in our data set    ex: In the validation_set,  num_of_examples = 1000
num_of_features = X.shape[1]                                                # The size of our X vector/Y is how many features we have in our data set

print('num_of_examples = ', num_of_examples)
print('num_of_features = ', num_of_features)

start = timeit.default_timer()                                              # Check runTime for Least Sqaures

# Least Sqaures Estimate Algorithm -----------------------------------------

XT = X.T                                                                    # X_transposed

XTdotX = XT.dot(X)                                                          # X_transposed * X

XTdotY = XT.dot(Y)                                                          # (XT * Y)

XTdotX_inv = np.linalg.inv(XTdotX)                                          # (XT * X)^-1

w = XTdotX_inv.dot(XTdotY)                                                  # (XT * X)^-1 * (XT * Y)

#print("w vector = ", w)                                                     # w is now the least squares estimate vector for any given example,

stop = timeit.default_timer()

print('Least Squares runTime: ', stop - start)


# Least Squares estimate complete!


# GRADIENT DESCENT   ------------------------------------------------------

epsilon = 0.00001

learning_rate = 0.01

init_learning_rate = 0.01

speed_of_decay = 0.01

alpha = (init_learning_rate/(1 + speed_of_decay))

alpha = 0.000001                                                            # temporarily set alpha to 0.0001

weights = np.zeros([num_of_features,1])                                     # Initial weight vector arbitrarily initalized to all 1s

l2_norm = 1                                                                 # l2_norm initialized to 1. (Anything greater than epsilon would be fine)


                                                                            # GRADIENT DESCENT FORMULA:       Wi  =  W(i-1) - 2 * alpha (XT*X - XT*Y)
print('epsilon = ', epsilon)
print('alpha = ', alpha)

start = timeit.default_timer()                                              # Check Gradient Descent runTime (XTdotX and XTdotY are already calculated though)

while(l2_norm > epsilon):

    oldWeights = weights                                                    # store the weights in a variable called oldWeights because we will subtract from new weights.
    XTXw = XTdotX.dot(weights)                                              # (XT * X) * w
    error = np.subtract(XTXw, XTdotY)                                       # ((XT * X) * w) - (XT * Y)
    norm_error = np.multiply(alpha,error)                                   # alpha * ((XT * X) * w) - (XT * Y)         referred to as norm_error

    weights = np.subtract(weights, norm_error)                              # Wi = W(i-1) - norm_error
    diff = np.subtract(weights, oldWeights)                                 # Wi - W(i-1)

    l2_norm = np.linalg.norm(diff, 'fro')

stop = timeit.default_timer()

print('Gradient Descent runTime: ', stop - start)

# Gradient Descent Complete


# We will now get the predicted values for the Least Sqaures solution and Gradient Descent respectively
# We will now get the predicted value for each example by multiplying the w vector by each example's features

# Least Squares predicted values

predicted_values = np.zeros(num_of_examples)                                # initialize array for predicted values for every example in validation set

observed_values = Y                                                         # Store Y in an array called observed_values (changing the name Y to observed values for readability)

for j in range(num_of_examples):                                            # For loop to multiply each weight by its corresponding feature and sum the products
    t = 0
    for i in range(num_of_features):                                        # Wo * feature1 + W1 * feature2 + .... = a predicted value for that example

        t = t + (w[i] * X[j][i])
    predicted_values[j] = t                                                 # Store the predicated values in predicted_values array



# Gradient Descent predicted values

predicted_values_GD = np.zeros(num_of_examples)                             # initialize predicted_values_GD array with zeros

for j in range(num_of_examples):                                            # For loop to multiply each weight by its corresponding feature and sum the products
    t = 0
    for i in range(num_of_features):                                        # Wo * feature1 + W1 * feature2 + .... = a predicted value for that example

        t = t + (weights[i] * X[j][i])
    predicted_values_GD[j] = t                                              # Store the predicated values in predicted_values_GD array



# Error Evaluation, we will now check the MSE, (R)MSE and MAE for our predicted values for Least Squares and Gradient Descent respectively

print('Least Sqaures Error Evaluation')

sumError = 0

for i in range(num_of_examples):                                            # Mean Squared Error  (This is the one we are asked to use to report performance!)
    diff = predicted_values[i] - observed_values[i]
    temp = np.square(diff)
    sumError = sumError + temp

MSE = sumError/num_of_examples                                              # MSE
print('MSE  = ', MSE)

sumError = 0
for i in range(num_of_examples):                                            # Root Mean Squared Error
    diff = predicted_values[i] - observed_values[i]
    temp = np.square(diff)
    temp = np.sqrt(temp)
    sumError = sumError + temp

RMSE = sumError/num_of_examples                                             #(R)MSE
print('RMSE = ', RMSE)

sumError = 0
for i in range(num_of_examples):                                            # Mean Absolute Error
    diff = predicted_values[i] - observed_values[i]
    temp = np.abs(temp)
    sumError = sumError + temp

MAE = sumError/num_of_examples                                              # MAE
print('MAE = ', MAE)



print('Gradient Descent Error Evaluation')

sumError = 0
for i in range(num_of_examples):                                            # Mean Squared Error  (This is the one we are asked to use to report performance!)
    diff = predicted_values_GD[i] - observed_values[i]
    temp = np.square(diff)
    sumError = sumError + temp

MSE = sumError/num_of_examples                                              # MSE
print('MSE  = ', MSE)

sumError = 0
for i in range(num_of_examples):                                            # Root Mean Squared Error
    diff = predicted_values_GD[i] - observed_values[i]
    temp = np.square(diff)
    temp = np.sqrt(temp)
    sumError = sumError + temp

RMSE = sumError/num_of_examples                                             # (R)MSE
print('RMSE = ', RMSE)

sumError = 0
for i in range(num_of_examples):                                            # Mean Absolute Error
    diff = predicted_values_GD[i] - observed_values[i]
    temp = np.abs(temp)
    sumError = sumError + temp

MAE = sumError/num_of_examples                                              # MAE
print('MAE = ', MAE)


# Lets plot some shit!

newfile = 'training_MSE.pk'

with open(newfile, 'rb') as fi:
    trainingerror = pk.load(fi)

newfile2 = 'test_MSE.pk'

with open(newfile2, 'rb') as pi:
    testerror = pk.load(pi)

pp.title('Number of Text Features vs MSE')
pp.xlabel('Number of text features')
pp.ylabel('Mean Squared Error')
pp.plot(range(160), trainingerror, '-r', label='Training MSE')
pp.plot(range(160), testerror, '-b', label='Test MSE')
pp.legend(loc='upper right')
pp.show()


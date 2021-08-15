import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from decimal import Decimal

# Load the data

def load_data():
    print("Loading data...")
    
    ham_files_location = os.listdir("dataset/ham")
    spam_files_location = os.listdir("dataset/spam")
    data = []
    
    # Load ham email
    for file_path in ham_files_location:
        f = open("dataset/ham/" + file_path, "r")
        text = str(f.read())
        data.append([text, "ham"])
    
    # Load spam email
    for file_path in spam_files_location:
        f = open("dataset/spam/" + file_path, "r")
        text = str(f.read())
        data.append([text, "spam"])
        
    data = np.array(data)
    
    print("flag 1: loaded data")
    return data


# Preprocessing data: punctuation and stopwords removal

def preprocess_data(data):
    print("Preprocessing data...")
    
    punc = string.punctuation           # Punctuation list
    sw = stopwords.words('english')     # Stopwords list
    
    for record in data:
        # Remove common punctuation and symbols
        for item in punc:
            record[0] = record[0].replace(item, "")
            
        # Lowercase all letters and remove stopwords 
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word  # Takes back all non-stopwords
        record[0] = newText
        
    print("flag 2: preprocessed data")        
    return data


# Splitting original dataset into training dataset and test dataset

def split_data(data):
    print("Splitting data...")
    
    features = data[:, 0]   # array containing all email text bodies
    labels = data[:, 1]     # array containing all corresponding labels
    
    training_data, test_data, training_labels, test_labels = train_test_split(features, labels, test_size = 0.27, random_state = 42)
    
    print("flag 3: splitted data")
    return training_data, test_data, training_labels, test_labels


# Returns count of each word in ham emails, count of each word in spam emails, count of ham emails, and count of spam emails

def get_count(training_data, training_labels):
    ham_dict = dict()   # dict of words that occurred in ham email and their corresponding frequency
    spam_dict = dict()  # dict of words that occurred in spam email and their corresponding frequency
    num_ham_emails = 0  # total number of ham emails
    num_spam_emails = 0 # total number of spam emails
    
    for i in range(len(training_data)):
        if training_labels[i] == "ham":
            num_ham_emails += 1
            for word in training_data[i].split():
                if word in ham_dict:
                    ham_dict[word] += 1
                else:
                    ham_dict[word] = 1
        else:
            num_spam_emails += 1
            for word in training_data[i].split():
                if word in spam_dict:
                    spam_dict[word] += 1
                else:
                    spam_dict[word] = 1
   
    print("flag 4: Get counts")

    return ham_dict, spam_dict, num_ham_emails, num_spam_emails


# Get likelihood p(word|class) of a word using smoothed likelihood function where n = total number of distinct words in a class, and alpha = 1

def get_likelihood(word, dictionary, total_num_words, n):
    alpha = 1
    
    if word in dictionary:
        likelihood = (dictionary[word] + alpha) / (total_num_words + n * alpha)   # total_num_words includes repetitive words
        return likelihood
    else:
        dictionary[word] = 0
        likelihood = (dictionary[word] + alpha) / (total_num_words + n * alpha)   # total_num_words includes repetitive words
        return likelihood


# Returns the class to which current email belongs to as a string

def get_class(p_ham, p_spam):
    if p_ham > p_spam:
        return "ham"
    else:
        return "spam"

# Multinomial Naive Bayes classifier, returns list of labels of test_data

def multinomialNB_classifier(training_data, training_labels, test_data, tsize):
    print("Running Multinomial NB Classifier...")
    result = []
    counter = 1
    
    # TRAINING STAGE
    
    # Getting the count of each word in ham emails, count of each word in spam emails, count of ham emails, and count of spam emails
    ham_dict, spam_dict, num_ham_emails, num_spam_emails = get_count(training_data, training_labels)
    
    num_words_ham = sum(list(ham_dict.values()))    # Number of words in ham
    num_words_spam = sum(list(spam_dict.values()))  # Number of words in spam
    
    n_ham_dict = len(ham_dict)      # number of distinct words in ham dictionary
    n_spam_dict = len(spam_dict)    # number of distinct words in spam dictionary
          
    # prior probability of ham email and spam email
    prior_ham = Decimal(num_ham_emails / (num_ham_emails + num_spam_emails))
    prior_spam = Decimal(num_spam_emails / (num_ham_emails + num_spam_emails))        
    
    # TESTING STAGE    
    for text in test_data:
        p_word_ham_product = Decimal(1)     # to calculate product operator on probability p(word|ham)
        p_word_spam_product  = Decimal(1)    # to calculate product operator on probability p(word|spam)
        
        for word in text.split():
            p_word_ham_product *= Decimal(get_likelihood(word, ham_dict, num_words_ham, n_ham_dict))      # product operator on p(word|ham)
            p_word_spam_product *= Decimal(get_likelihood(word, spam_dict, num_words_spam, n_spam_dict)) # product operator on p(word|spam)
        
        # probability of email being ham or spam 
        p_ham = prior_ham * p_word_ham_product 
        p_spam = prior_spam * p_word_spam_product
        
        # Predicting the class of email
        result.append(get_class(p_ham, p_spam))
        
        print(str(counter) + "/" + str(tsize) + " done!")
        counter += 1
    
    return result
    
        
# main function

def main():
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    
    tsize = len(test_data) # sample size of test emails to be tested. Use len(test_data) to test all test_data
    
    result = multinomialNB_classifier(training_data, training_labels, test_data[:tsize], tsize)
    accuracy = accuracy_score(test_labels[:tsize], result)
    
    print("training data size\t: " + str(len(training_data)))
    print("test data size\t\t: " + str(len(test_data)))
    print("Samples tested\t\t: " + str(tsize))
    print("% accuracy\t\t\t: " + str(accuracy * 100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
    
main()
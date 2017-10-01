import numpy as np
import pandas as pd
from scipy import stats
import general
from sklearn import svm

# TODO: fill these variables according to your data files.
personality_participant_col_name = "id"     #name of participant column in personality_info file
data_participant_col_name = "Participant"   #name of participant column in data file


def find_correlations(data, personality_data):
    """This function creates a file containing a correlation matrix between defined interesting parameters"""
    combined = personality_data.merge(data, how='inner', left_on=personality_participant_col_name,
                                      right_on=data_participant_col_name)
    # Compute the correlation between every two numerical columns in the dataFrame and return a matrix
    res = combined.corr()
    # cut from matrix only interesting parts:
    res = res.loc[list(data), list(personality_data)]
    # TODO: define path and desired name for output file
    output_filename = "outputs/correlations matrix.csv"
    print "The correlation matrix is saved in file {}...".format(output_filename)
    res.to_csv(output_filename, encoding='latin1')


def run_t_tests(data, personality_data):
    """This function runs """
    # the results dataFrame will store data of all significant t-tests
    results = pd.DataFrame(columns=["eye movement trait", "personality trait", "p-value", "t-value",
                                    "low mean", "high mean", "low std", "high std"])
    results_index = 0
    # t-tests will run on all pairs of data-personality columns except from participants id:
    personality_categories = list(personality_data)
    personality_categories.remove(personality_participant_col_name)
    data_categories = list(data)
    data_categories.remove(data_participant_col_name)

    for feature in data_categories:
        if data[feature].dtype != np.float64 and data[feature].dtype != np.int64:
            # skipping non numerical columns
            continue
        for trait in personality_categories:
            # get upper and lower third of sample according to the personality trait
            # TODO: percentage can be changed
            low, high = general.get_upper_lower_percentile(data, trait, 0.33)
            # run t-test to check if low and high samples are significantly different
            t, p = stats.ttest_ind(low[feature], high[feature])
            if p < 0.05:
                # if a significant difference was found, add it to results dataFrame:
                new_res = [feature, trait, p, t, low[feature].mean(), high[feature].mean(), low[feature].std(),
                           high[feature].std()]
                results.loc[results_index] = new_res
                results_index += 1

    # TODO: define path and desired name for output file
    output_filename = "outputs/t-tests_results.csv"
    print "Significant t-tests are saved in file {}...".format(output_filename)
    results.to_csv(output_filename, encoding='latin1')


def classify_by_trait(data, trait, num_iterations):
    """"""
    data_copy = data.dropna(axis=0)
    data_copy['label'] = ""
    sum_accuracy = 0
    # define the label for each row in data according to the personality trait
    data_copy = fill_label_column(data_copy, trait, 'label')
    # remove from data all entries who got middle label, leaving only high/low labels in data:
    labeled_data = data_copy[data_copy['label'] != 'middle']
    for i in range(num_iterations):
        # split data into random training and test set
        training_set, test_set = general.split_df_to_two(labeled_data, 0.8)
        classifier = build_classifier(training_set)
        sum_accuracy += assess_classifier_accuracy(classifier, test_set)
        if i % 10 == 0:
            print "Computing classifier {}/{} for trait {}.......".format(i, num_iterations, trait)
            print "Average classification accuracy so far is {}".format(sum_accuracy/(i + 1))
    print "Across all classifiers, predicting {} according to data is accurate in {}% of time".format(trait,

                                                                                                      (sum_accuracy / num_iterations)*100)
def fill_label_column(df, trait, label_col_name):
    """This function finds the column named <label_col_name> in the df and fills it with a label to each row
     according to the given personality trait. The label is given so that upper third of sample is labeled 1,
     middle third is labeled 0 and low third is labeled -1. percentage can be changed."""
    # TODO: if you want the labels to be defined by another percentage, change here
    low_participants, high_participants = general.get_upper_lower_percentile_list(trait, 0.33)
    for index, row in df.iterrows():
        if str(row[data_participant_col_name]) in low_participants:
            df.loc[index, label_col_name] = 'low'
        elif str(row[data_participant_col_name]) in high_participants:
            df.loc[index, label_col_name] = 'high'
        else:
            df.loc[index, label_col_name] = 'middle'
    return df


def build_classifier(training_set):
    # input for classier: X - data vectors, y - labels
    training_set_copy = training_set.drop('label', axis=1)
    X = training_set_copy.drop(data_participant_col_name, axis=1).as_matrix()
    # TODO:  if you want to know the size of X, print X.shape
    y = np.asarray(training_set['label'], dtype="|S6")

    # compute the classifier function according to training set
    classifier = svm.SVC(kernel='linear', C=1000)
    return classifier.fit(X, y)


def assess_classifier_accuracy(classifier, labeled_test_set):
    """This function runs the classifier on the test set and returns the percent of times it predicted a
    participant accurately"""
    data_categories = list(labeled_test_set)
    data_categories.remove('label')
    data_categories.remove(data_participant_col_name)
    labeled_test_set["Prediction"] = ""
    prediction_accuracy_sum = 0
    # iterating over the test set and predicting label for each row according to classifier
    for index, row in labeled_test_set.iterrows():
        point = row[data_categories].values.reshape(1, -1)
        prediction = classifier.predict(point)[0]
        labeled_test_set.loc[index, 'Prediction'] = prediction
        if prediction == row['label']:
            prediction_accuracy_sum += 1
    # if test set contains several lines for each participant, we compute final prediction of participant according
    # to majority of its labels
    grouped_prediction = labeled_test_set[[data_participant_col_name, 'label', 'Prediction']].\
        groupby(data_participant_col_name).agg(
        lambda x: x.value_counts().index[0])
    total_accuracy_sum = 0
    # TODO: currently, there is no use to these values. If you want to use them, you have to implement the usage
    high_as_low = 0     # an index that counts how many times a 'high' label got 'low' prediction
    low_as_high = 0     # an index that counts how many times a 'low' label got 'high' prediction
    for index, row in grouped_prediction.iterrows():
        if row['label'] == row['Prediction']:
            total_accuracy_sum += 1
        else:
            if row['label'] == 'high':
                high_as_low += 1
            else:
                low_as_high += 1
    return float(total_accuracy_sum) / grouped_prediction.shape[0]


if __name__ == '__main__':
    # load data for correlation test:
    trial_data = general.load_and_clean_data('../raw_data/Event Statistics - Trial Summary.txt')
    personality_data = general.read_personality_data()
    # Grouping the data by participant:
    data_for_ttest = trial_data.groupby(data_participant_col_name).mean().reset_index()
    # find_correlations(data_for_correlations, personality_data)

    data_for_t_test = ""
    run_t_tests(data_for_ttest, personality_data)
    data_for_classifier = ""

    # find_correlations(data, personality_data)
    # data = general.load_and_clean_data('../raw_data/Event Statistics - Trial Summary.txt')
    # data = data[['Participant', 'Fixation Count', 'Saccade Count']]
    # classify_by_trait(data, 'E', 400)

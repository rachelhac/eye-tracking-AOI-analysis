import pandas as pd
import numpy as np

# TODO: fill these variables according to your data files.
personality_participant_col_name = "id"
data_participant_col_name = "Participant"
overview_participant_col_name = "Participant"       #participant column name in the overview file
personality_data_filename = '../raw_data/partcipants_info1.txt'
personality_file_separator = "\t"
participants_overview_filename = '../raw_data/Specialized Statistics - Participant Overview.txt'
participants_overview_separator = "\t"
# TODO: insert here the relative path of the directory you want the output files to be stored at.
output_directory = 'outputs'


def read_personality_data():
    """This function returns a dataFrame containing the personality data of all valid participants.
    It excludes non native Hebrew speakers and noon valid participants according to
     get_valid_tracking_participants_list() function"""
    personality_data = pd.read_csv(personality_data_filename, sep=personality_file_separator,
                                   error_bad_lines=False)
    types_dict = {personality_participant_col_name: 'str'}
    personality_data = personality_data.astype(types_dict, errors='ignore')
    valids = get_valid_tracking_participants_list()
    personality_data = personality_data[personality_data[personality_participant_col_name].isin(valids)]
    # TODO: make sure that this is relevant in your data:
    personality_data = personality_data[personality_data["Culture"] == 1]
    del personality_data['Culture']
    return personality_data


def get_valid_tracking_participants_list():
    """This function sorts out the participants that are not valid. In this experiment we excluded participans
    with tracking ratio smaller than 84%"""
    # remove low tracking ratio
    participants_data = pd.read_csv(participants_overview_filename, sep=participants_overview_separator,
                                    encoding='latin1', error_bad_lines=False)
    types_dict = {data_participant_col_name: 'str'}
    participants_data = participants_data.astype(types_dict, errors='ignore')
    participants_data = participants_data[participants_data['Tracking Ratio [%]'] > 84]
    valid_participants_tracking = set(participants_data[overview_participant_col_name])
    return valid_participants_tracking


def get_valid_participants_list():
    personality_data = read_personality_data()
    return list(personality_data[personality_participant_col_name].unique())

def load_and_clean_data(filename):
    """creates the dataframe object from file and does some cleaning. It is good for Trial and AOI
    single and summary files. More cleaning can be added here"""
    # Assuming separator is TAB in all files except csv files.
    separator = "\t"
    if filename.endswith("csv"):
        separator = ","
    data = pd.read_csv(filename, sep=separator, encoding='latin1', low_memory=False)
    # remove left eye data:
    if 'Eye L/R' in list(data.columns.values):
        data = data[data['Eye L/R'] == 'Right']
    # remove data of excluded participants:
    participants_to_include = get_valid_participants_list()
    print "Found {} valid participants in file {}".format(len(participants_to_include), filename)
    data = data[data[data_participant_col_name].isin(participants_to_include)]
    # remove all non informative trials
    data = data[data['Stimulus'] != 'fix_mid_down.png']
    data = data[data['Stimulus'] != 'Validation.jpg']
    data = data[data['Stimulus'] != 'answer_the_question.png']
    data = data[data['Stimulus'] != 'intro_interview_part.png']
    # in AOI files - fix some typos in aoi names
    # TODO: if you have other typos in your file, you can fix them like that
    if 'AOI Name' in list(data.columns.values):
        data['AOI Name'] = data['AOI Name'].apply(lambda x: 'eyes+eyebrows' if x == 'eyes+etebrows' else x)
        data = data[data['AOI Name'] != 'AOI 001']
    return data


def get_upper_lower_percentile(data_frame, big_5_trait, percentile, lower_output_filename="", upper_output_filename="",):
    """This function splits the dataframe into 2 smaller dataframes and returns them. The first one is a dataFrame
     that contains the data of the participants who got the percentile lowest level of the personality
     trait (E/A/C/N/I) and the second of participants who got the highest level of that trait."""
    lower_participants, higher_participants = get_upper_lower_percentile_list(big_5_trait, percentile)
    higher_data = data_frame[data_frame[data_participant_col_name].isin(higher_participants)]
    lower_data = data_frame[data_frame[data_participant_col_name].isin(lower_participants)]
    if lower_output_filename:
        print "file created {}/{}".format(output_directory, lower_output_filename)
        lower_data.to_csv(output_directory + "/" + lower_output_filename)
    if upper_output_filename:
        print "file created {}/{}".format(output_directory, upper_output_filename)
        higher_data.to_csv(output_directory + "/" + upper_output_filename)
    return lower_data, higher_data


def get_upper_lower_percentile_list(big_5_trait, percentile):
    """returns two lists of participants - low and high in the given trait according to the given percentile"""
    # get the personality data and remove non valid ones
    personality_data = read_personality_data()
    participants_to_include = get_valid_participants_list()
    personality_data = personality_data[personality_data[personality_participant_col_name].isin(participants_to_include)]

    lower_third_threshold = personality_data[big_5_trait].quantile(percentile)
    # TODO: if you want to know what is the threshold, uncomment these lines
    # print "low {} threshold is {}".format(big_5_trait, lower_third_threshold)
    higher_third_threshold = personality_data[big_5_trait].quantile(1 - percentile)
    # print "high {} threshold is {}".format(big_5_trait, higher_third_threshold)
    lower_participants = []
    higher_participants = []
    for index, row in personality_data.iterrows():
        if row[big_5_trait] <= lower_third_threshold:
            lower_participants.append(str(int(row[personality_participant_col_name])))
        elif row[big_5_trait] > higher_third_threshold:
            higher_participants.append(str(int(row[personality_participant_col_name])))
    return lower_participants, higher_participants


def split_df_to_two(df, split_fraction=0.5, df1_output_filename="", df2_output_filename=""):
    """" This function randomly splits the dataFrame df into 2 parts with sizes according to split_fraction.
    It splits the data according to participants so that one participant's data isn't split.
    By default, split_fraction=0.5, which divides df into 2 equal parts."""
    new_df = df.reset_index(drop=True)
    participants = df[data_participant_col_name].unique()
    partition1 = list(np.random.choice(participants, size=int(len(participants)*split_fraction), replace=False))
    partition2 = [x for x in participants if x not in partition1]
    df1 = new_df[new_df[data_participant_col_name].isin(partition1)]
    df2 = new_df[new_df[data_participant_col_name].isin(partition2)]
    if df1_output_filename:
        print "file created {}/{}".format(output_directory, df1_output_filename)
        df1.to_csv(output_directory + "/" + df1_output_filename)
    if df2_output_filename:
        print "file created {}/{}".format(output_directory, df2_output_filename)
        df2.to_csv(output_directory + "/" + df2_output_filename)
    return df1, df2

def split_df_by_beginning_end_of_movie(df, begin_output_filename="", end_output_filename=""):
    """returns """
    beginning = df[df['Stimulus'].str.match(".*_q1.*")]
    end = df[df['Stimulus'].str.match(".*_q2.*")]
    if begin_output_filename:
        print "file created {}/{}".format(output_directory, begin_output_filename)
        beginning.to_csv(output_directory + "/" + begin_output_filename)
    if end_output_filename:
        print "file created {}/{}".format(output_directory, end_output_filename)
        end.to_csv(output_directory + "/" + end_output_filename)
    return beginning, end

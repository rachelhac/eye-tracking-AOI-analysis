import pandas as pd
import stat_analysis
from scipy import stats
import general

# TODO: insert here the relative path of the directory you want the output files to be stored at.
output_directory = 'outputs'


def general_analysis(aoi_summary, output_filename=""):
    """A simple function that prints some basic overview on the data, grouped by AOIs across all participants."""
    # calculate the average number and time of fixations on each region.
    # TODO: make sure that col names match your data. You can put other col names to your interest
    aoi_fixations_count = aoi_summary[['AOI Name', 'Fixation Count', 'Fixation Time [%]']].groupby('AOI Name').mean()
    aoi_fixations_count.sort_values('Fixation Count', ascending=False, inplace=True)
    aoi_fixations_count = aoi_fixations_count[aoi_fixations_count.index != 'White Space']
    print aoi_fixations_count
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        aoi_fixations_count.to_csv(output_directory + "/" + output_filename)


def general_order_of_regions(aoi_single, output_filename=""):
    """A general function that gives some overview on the data. It shows, for each AOI, how many fixations it
    got as first fixation point, second, etc.
    """
    orders = aoi_single[['AOI Name', 'Index']]
    res = orders.groupby(['Index', 'AOI Name']).size()
    print res
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        res.to_csv(output_directory + "/" + output_filename)


def compute_trial_pattern(aoi_single, regions_apart=False, output_filename=""):
    """This function computes to each participant and to each trial, what was the pattern of AOIs fixations
    in the trial. It returns a new dataframe that summarizes this data.
    the flag region_apart defines if each region will appear alone in the pattern or with the previous regions"""
    print "computing patterns....."
    aoi_patterns = pd.DataFrame(columns=["Participant", "Stimulus"])
    orders = aoi_single[['Participant', 'Stimulus', 'AOI Name', 'Index']].dropna(axis='index')
    grouped = orders.groupby(['Participant', 'Stimulus'])
    for key, item in grouped:
        pattern = []
        index = 1
        while len(pattern) < 3 and index < 10:
            aoi_name = grouped.get_group(key)[grouped.get_group(key)['Index'] == index].reset_index(drop=True)
            if not aoi_name.empty:
                if aoi_name.loc[0, 'AOI Name'] == "neck" and len(pattern) == 0:
                    index += 1
                    continue
                if len(pattern) > 0 and aoi_name.loc[0, 'AOI Name'] == pattern[-1]:
                    index += 1
                    continue
                pattern.append(aoi_name.loc[0, 'AOI Name'])
            index += 1
        pattern1 = pattern[0] if len(pattern) > 0 else "none"
        pattern2 = pattern[1] if len(pattern) > 1 else "none"
        pattern3 = pattern[2] if len(pattern) > 2 else "none"
        if regions_apart:
            aoi_patterns = aoi_patterns.append({
                "Participant": key[0],
                "Stimulus": key[1],
                "1st Region": pattern1,
                "2nd Region": pattern2,
                "3rd Region": pattern3,
                "length": len(pattern)
            }, ignore_index=True)
        else:
            aoi_patterns = aoi_patterns.append({
                "Participant": key[0],
                "Stimulus": key[1],
                "1st Region": pattern1,
                "2nd Region": "{}, {}".format(pattern1, pattern2),
                "3rd Region": "{}: {}, {}, {}".format(len(pattern), pattern1, pattern2, pattern3),
                "length": len(pattern)
                }, ignore_index=True)
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        aoi_patterns.to_csv(output_directory + "/" + output_filename)
    return aoi_patterns


def general_patterns_overview(patterns_single, output_filename=""):
    df = patterns_single.groupby('1st Region').size().reset_index().rename(columns={'1st Region': 'Region', 0: 'count'})
    df.sort_values('count', ascending=False, inplace=True)
    df['Region'] = df['Region'].astype(str) + " 1st"
    new = df
    df = patterns_single.groupby('2nd Region').size().reset_index().rename(columns={'2nd Region': 'Region', 0: 'count'})
    df.sort_values('count', ascending=False, inplace=True)
    df['Region'] = df['Region'].astype(str) + " 2nd"
    new = new.append(df)
    df = patterns_single.groupby('3rd Region').size().reset_index().rename(columns={'3rd Region': 'Region', 0: 'count'})
    df.sort_values('count', ascending=False, inplace=True)
    df['Region'] = df['Region'].astype(str) + " 3rd"
    new = new.append(df)
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        new.to_csv(output_directory + "/" + output_filename)
    return new

def count_participant_patterns(aoi_patterns, pattern_len_col, output_filename=""):
    """This function receives as input the patterns dataframe (output of compute_trial_pattern() function)
    and counts for each participant, how many times s/he demonstrated each pattern.
    pattern_len_col defines what patterns we want to count - 1st Region, 2nd or 3rd"""

    count = aoi_patterns.groupby(['Participant', pattern_len_col]).size().reset_index().rename(columns={0:'count'})
    # all rest of function is to reshape count to a dataframe form:
    regions = list(count[pattern_len_col].unique())
    new_count = pd.DataFrame(columns=['Participant'] + regions)
    current_participant = count.loc[0, 'Participant']
    new_count_index = 0
    for index, row in count.iterrows():
        if row['Participant'] != current_participant:
            new_count_index += 1
            current_participant = row['Participant']
        new_count.loc[new_count_index, 'Participant'] = row['Participant']
        new_count.loc[new_count_index, row[pattern_len_col]] = row['count']
    new_count.fillna(value=0, inplace=True)
    new_count['Most common pattern'] = new_count[regions].idxmax(axis=1)
    new_count['Most common consistency'] = ""
    new_count['Most common consistency'] = new_count.apply(lambda row: float(max(row[regions]))/sum(row[regions]), axis=1)
    new_count['General consistency'] = ""
    new_count['General consistency'] = new_count.apply(lambda row: row[regions].std(), axis=1)
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        new_count.to_csv(output_directory + "/" + output_filename)
    return new_count


def get_patterns_length_count(patterns, output_filename=""):
    df = patterns[['Participant', 'length']].groupby(['Participant', 'length']).size().reset_index().rename(columns={0: 'count'})
    df1 = df[df['length'] == 1].rename(columns={'count': 'count_len1'})
    df1 = df1.drop('length', 1)
    df2 = df[df['length'] == 2].rename(columns={'count': 'count_len2'})
    df2 = df2.drop('length', 1)
    df3 = df[df['length'] == 3].rename(columns={'count': 'count_len3'})
    df3 = df3.drop('length', 1)
    res = df1.merge(df2, on='Participant')
    res = res.merge(df3, on='Participant')
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        res.to_csv(output_directory + "/" + output_filename)
    return res




def build_big_patterns_counts(aoi_single, output_filename=""):
    """This function builds a big dataframe that counts for each participant, all possible patterns of all
    3 lengths."""
    patterns = compute_trial_pattern(aoi_single)
    # big = count_participant_patterns(patterns, '1st Region')
    # pattern2 = count_participant_patterns(patterns, '2nd Region')
    big = count_participant_patterns(patterns, '3rd Region')
    # big = big.merge(pattern2, on='Participant')
    # big = big.merge(pattern3, on='Participant')
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        big.to_csv(output_directory + "/" + output_filename)
    return big


def group_summary_by_AOIs(aoi_summary, output_filename=""):
    """This function takes some of the data from aoi_summary grouped by AOIs and rearrange it in columns"""
    aois = list(aoi_summary['AOI Name'].unique())
    df = aoi_summary[['Participant']].groupby('Participant').agg('count').reset_index()
    #TODO: you choose here other columns from aoi_summary
    columns_of_interest = ['Fixation Count', 'Fixation Time [%]', 'Average Fixation Duration [ms]']
    for region in aois:
        temp = aoi_summary[aoi_summary['AOI Name'] == region]
        temp = temp[['Participant', 'Fixation Count', 'Fixation Time [%]', 'Average Fixation Duration [ms]']]
        temp = temp.rename(columns={'Fixation Count': (region + ' Fixation Count'), 'Fixation Time [%]': (region +
                                ' Fixation Time [%]'), 'Average Fixation Duration [ms]': (region +
                                ' Average Fixation Duration [ms]')})
        temp = temp.groupby('Participant').mean().reset_index()
        df = df.merge(temp, on='Participant')
    if output_filename:
        print "file created {}/{}".format(output_directory, output_filename)
        df.to_csv(output_directory + "/" + output_filename)
    return df


if __name__ == '__main__':
    aoi_summary = general.load_and_clean_data('../raw_data/AOI Statistics - Trial Summary (AOI).txt')
    aoi_single = general.load_and_clean_data('../raw_data/AOI Statistics - Single.txt')
    personality_data = general.read_personality_data()
    # 1. sum of all fixations on all AOIs:
    # general_analysis(aoi_summary)
    # 2. summarize general orders of AOIs
    # general_order_of_regions(aoi_single)
    # 3. compute patterns and get general overview:
    patterns_single = compute_trial_pattern(aoi_single, regions_apart=True, output_filename="patterns_single.csv" )
    general_patterns_overview(patterns_single, "patterns_overview.csv")
    # 4. get patterns count:
    patterns = compute_trial_pattern(aoi_single, output_filename="patterns.csv")        # find pattern of each trial
    pat_len = get_patterns_length_count(patterns)
    stat_analysis.find_correlations(pat_len, personality_data, "len-personality correlations.csv")
    stat_analysis.run_t_tests(pat_len, personality_data, "len-personality t-tests.csv")
    begin, end = general.split_df_by_beginning_end_of_movie(patterns)
    patterns_count_len2 = count_participant_patterns(patterns, '2nd Region', output_filename="patterns_count_len2.csv")
    patterns_count_len2_q1 = count_participant_patterns(begin, '2nd Region', output_filename="patterns_count_len2_q1.csv")
    patterns_count_len2_q2 = count_participant_patterns(end, '2nd Region', output_filename="patterns_count_len2_q2.csv")
    # 5. run statistical tests on data:
    stat_analysis.find_correlations(patterns_count_len2, personality_data, "len2-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len2, personality_data, "len2-personality t-tests.csv")
    stat_analysis.find_correlations(patterns_count_len2_q1, personality_data, "len2_q1-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len2_q1, personality_data, "len2_q1-personality t-tests.csv")
    stat_analysis.find_correlations(patterns_count_len2_q2, personality_data, "len2_q2-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len2_q2, personality_data, "len2_q2-personality t-tests.csv")
    # 6. if you want all patterns_count grouped together:
    # big = build_big_patterns_counts(aoi_single, output_filename="patterns_count_join.csv")
    # 7. analyze data about first region only:
    patterns_count_len1 = count_participant_patterns(patterns, '1st Region', output_filename="patterns_count_len1.csv")
    patterns_count_len1_q1 = count_participant_patterns(begin, '1st Region', output_filename="patterns_count_len1_q1.csv")
    patterns_count_len1_q2 = count_participant_patterns(end, '1st Region', output_filename="patterns_count_len1_q2.csv")
    # 8. run statistical tests on data:
    stat_analysis.find_correlations(patterns_count_len1, personality_data, "len1-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len1, personality_data, "len1-personality t-tests.csv")
    stat_analysis.find_correlations(patterns_count_len1_q1, personality_data, "len1_q1-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len1_q1, personality_data, "len1_q1-personality t-tests.csv")
    stat_analysis.find_correlations(patterns_count_len1_q2, personality_data, "len1_q2-personality correlations.csv")
    stat_analysis.run_t_tests(patterns_count_len2_q2, personality_data, "len1_q2-personality t-tests.csv")

    # 7. get more data about participants across AOIs:
    data_by_AOI = group_summary_by_AOIs(aoi_summary)
    stat_analysis.find_correlations(data_by_AOI, personality_data, "AOIs-general-personality correlations.csv")
    stat_analysis.run_t_tests(data_by_AOI, personality_data, "AOIs-general-personality t-tests.csv")
    # 8. run classifier on data:
    data_for_classifier = patterns_count_len1[['Participant', 'eyes+eyebrows']]
    stat_analysis.classify_by_trait(data_for_classifier, 'E', 400)
    data_for_classifier = patterns_count_len1[['Participant', 'eyes+eyebrows', 'mouth+chin', 'nose+cheeks']]
    stat_analysis.classify_by_trait(data_for_classifier, 'E', 400)
    # data_for_classifier = pat_len
    # stat_analysis.classify_by_trait(data_for_classifier, 'E', 400)
    # 9. statistical analysis on raw data:
    stat_analysis.find_correlations(aoi_summary.groupby('Participant').mean().reset_index(), personality_data, "raw-correlations.csv")
    stat_analysis.run_t_tests(aoi_summary.groupby('Participant').mean().reset_index(), personality_data, "raw-t-tests.csv")





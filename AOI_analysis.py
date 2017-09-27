import pandas as pd
import stat_analysis
from scipy import stats
import general


def general_analysis(aoi_summary):
    """A simple function that prints some basic overview on the data, grouped by AOIs across all participants."""
    # calculate the average number and time of fixations on each region.
    # TODO: make sure that col names match your data. You can put other col names to your interest
    aoi_fixations_count = aoi_summary[['AOI Name', 'Fixation Count', 'Fixation Time [%]']].groupby('AOI Name').mean()
    aoi_fixations_count.sort_values('Fixation Count', ascending=False, inplace=True)
    aoi_fixations_count = aoi_fixations_count[aoi_fixations_count.index != 'White Space']
    print aoi_fixations_count
    # TODO: If you want to export this data, uncomment the following line:
    # aoi_fixations_count.to_csv('outputs/general_overview.csv')


def general_order_of_regions(aoi_single):
    """A general function that gives some overview on the data. It shows, for each AOI, how many fixations it
    got as first fixation point, second, etc.
    """
    orders = aoi_single[['AOI Name', 'Index']]
    res = orders.groupby(['Index', 'AOI Name']).size()
    print res
    # TODO: If you want to export this data, uncomment the following line:
    res.to_csv('outputs/general_order.csv')


def compute_trial_pattern(aoi_single):
    """This function computes to each participant and to each trial, what was the pattern of AOIs fixations
    in the trial. It returns a new dataframe that summarizes this data"""
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
        pattern1 = pattern[0] if len(pattern) > 0 else ""
        pattern2 = pattern[1] if len(pattern) > 1 else ""
        pattern3 = pattern[2] if len(pattern) > 2 else ""
        aoi_patterns = aoi_patterns.append({
            "Participant": key[0],
            "Stimulus": key[1],
            "1st Region": pattern1,
            "2nd Region": "{}, {}".format(pattern1, pattern2),
            # "2nd Region": pattern2,   #TODO: if you want to store only the second region and not the whole pattern
            "3rd Region": "{}, {}, {}".format(pattern1, pattern2, pattern3)
            # "3rd Region": pattern3
            }, ignore_index=True)
    # aoi_patterns.to_csv('outputs/patterns.csv')
    return aoi_patterns

def count_participant_patterns(aoi_patterns, pattern_len_col):
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
    new_count['Stability'] = ""
    new_count['Stability'] = new_count.apply(lambda row: row[regions].std(), axis=1)
    return new_count


def build_big_patterns_counts(aoi_single):
    """This function builds a big dataframe that counts for each participant, all possible patterns of all
    3 lengths."""
    patterns = compute_trial_pattern(aoi_single)
    big = count_participant_patterns(patterns, '1st Region')
    pattern2 = count_participant_patterns(patterns, '2nd Region')
    # pattern3 = count_participant_patterns(patterns, '3rd Region')
    big = big.merge(pattern2, on='Participant')
    # big = big.merge(pattern3, on='Participant')
    return big


def group_summary_by_AOIs(aoi_summary):
    """This function takes some of the data from aoi_summary grouped by AOIs and rearrange it so that the data
    about each AOI is stored in a different column"""
    aois = list(aoi_summary['AOI Name'].unique())
    #TODO: you choose here other columns from aoi_summary
    columns_of_interest = ['Fixation Count', 'Fixation Time [%]', 'Average Fixation Duration [ms]']
    grouped = aoi_summary[['Participant', 'AOI Name'] + columns_of_interest].groupby(['Participant', 'AOI Name']).\
        mean().reset_index()
    print grouped
    # reshape data:
    df_cols = ["{} {}".format(i, j) for i in aois for j in columns_of_interest]
    print df_cols
    df = pd.DataFrame(columns=df_cols)
    df_index = 0
    for key, item in grouped.iterrows():
        print "key = {}".format(key)
        print "item = {}".format(item)
    # print df



def group_summary_by_AOIs2(aoi_summary):
    """This function takes some of the data from aoi_summary grouped by AOIs and rearrange it """
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
    # df.to_csv("outputs/bb.csv")
    # count_participant_patterns(aoi_single)
    print df
    return df

if __name__ == '__main__':
    aoi_summary = general.load_and_clean_data('../raw_data/AOI Statistics - Trial Summary (AOI).txt')
    aoi_single = general.load_and_clean_data('../raw_data/AOI Statistics - Single.txt')
    # 1. sum of all fixations on all AOIs:
    # general_analysis(aoi_summary)
    # 2. summarize general orders of AOIs
    # general_order_of_regions(aoi_single)
    # 3. get patterns count:
    patterns = compute_trial_pattern(aoi_single)
    patterns_count_len2 = count_participant_patterns(patterns, '2nd Region')
    patterns_count_len1 = count_participant_patterns(patterns, '1st Region')
    # 4. if you want all patterns_count grouped together:
    # big = build_big_patterns_counts(aoi_single)
    # 5. get more data about participants across AOIs:
    # data = group_summary_by_AOIs2(aoi_summary)
    # data = data.merge(patterns_count_len1, on='Participant')

    # 6. run statistical tests on data collected:
    personality = general.read_personality_data()
    stat_analysis.classify_by_trait(patterns_count_len1, 'E', 400)
    # stat_analysis.run_t_tests(patterns_count_len2, personality)
    # stat_analysis.find_correlations(data, personality)




    # data = analyze_aois(aoi_single, aoi_summary)

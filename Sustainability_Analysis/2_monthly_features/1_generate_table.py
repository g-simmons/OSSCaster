import json
from datetime import datetime
import os
from features import *
from scipy.stats import skew
from math import ceil
from tqdm import tqdm
import numpy as np 

with open('../end_date_dict.json', 'r') as f:
    end_time_dict = json.load(f)

with open('../start_date_dict.json', 'r') as f:
    start_time_dict = json.load(f)

c_path = '../monthly_data/commits/'
commits_p = os.listdir(c_path)

e_path = '../monthly_data/emails/'
emails_p = os.listdir(e_path)

# projects should have both commits and emails.
projects = set(commits_p).intersection(set(emails_p))
projects = sorted(list(projects))

################# process commits #################
def cal_tehnical_net(commit_path):

    with open(commit_path, 'r') as f:
        commits = f.readlines()

    commit_time_list = []
    commit_dict = {}
    file_set = set()

    for commit in commits:
        # append the commit time # 2005-12-14T15:36:16Z
        commit_time, author, file = commit.replace('\n', '').split(',')
        commit_time_list.append(commit_time)
        # add the file
        file_set.add(file)
        # add to the author-commit histroy
        if author not in commit_dict:
            commit_dict[author] = []
        commit_dict[author].append(commit_time)

    # num of files
    num_files = len(file_set)
    # num of commits
    num_commits = len(commits)
    # last commit to graduation
    commit_time_list = sorted(commit_time_list)

    # skewnees
    commit_time_list = [datetime.strptime(c, '%Y-%m-%d %H:%M:%S') for c in commit_time_list]
    commit_time_list = [(c - start_time_d).days / 30 for c in commit_time_list]
    skew_c = skew(commit_time_list)
    if np.isnan(skew_c): skew_c = 0

    # top3 of ratio of interuption time
    inactive_c = get_interrupt_coe_commit(commits)
    
    # core committers (top 10%)
    num_committers = len(commit_dict.keys())
    num_core_devs = ceil(0.1 * num_committers)
    # reverse sorting to get the top 10% commits by developers
    commits_dis = sorted([len(commit_times) for commit_times in list(commit_dict.values())], reverse=True)
    commits_by_core = commits_dis[:num_core_devs]
    if sum(commits_dis) != 0:
        c_percentage = sum(commits_by_core) / sum(commits_dis)

    # network features
    c_nodes, c_edges, c_c_coef, c_mean_degree = get_network_features_c(commits)
    c_long_tail = get_degree_longtail_c(commits)
    c_triangles = get_triangles_c(commits)

    technical_features = [num_commits, num_committers, num_files, skew_c, c_percentage, inactive_c, \
             c_nodes, c_edges, c_c_coef, c_mean_degree, c_long_tail, c_triangles]
             
    return technical_features

################# process emails #################
def cal_social_net(email_path):

    with open(email_path, 'r') as f:
        emails = f.readlines()

    email_time_list = []
    email_dict = {}
    respondent_set = set()
    
    for email in emails:

        email_time, sender, respondent = email.replace('\n', '').split(',')
        email_time_list.append(email_time)
        respondent_set.add(respondent)

        if sender not in email_dict:
            email_dict[sender] = []
        email_dict[sender].append(email_time)

    # num of emails
    num_emails = len(emails)

    # last email to graduation
    email_time_list = sorted(email_time_list)

    # skewnees
    email_time_list = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in email_time_list]
    email_time_list = [(e - start_time_d).days / 30 for e in email_time_list]
    skew_e = skew(email_time_list)
    if np.isnan(skew_e): skew_e = 0

    # interuption days
    inactive_e = get_interrupt_coe_email(emails)

    # senders and respondants
    num_senders = len(email_dict.keys())
    num_respondents = len(respondent_set)
    
    # core senders (top 10%)
    num_core_sender = ceil(0.1 * num_senders)
    emails_dis = sorted([len(times) for times in list(email_dict.values())], reverse=True)
    emails_by_core = emails_dis[:num_core_sender]
    if sum(emails_dis)!= 0:
        e_percentage = sum(emails_by_core) / sum(emails_dis)

    # network features
    network_features = get_network_features_e(emails)
    # print(len(network_features))
    e_nodes, e_edges, e_c_coef, e_mean_degree, e_bidirected_edges = network_features
    e_long_tail = get_degree_longtail_e(emails)
    e_triangles = get_triangles_e(emails)

    # e_communicatability = get_communicability(e_full_path)

    social_net_features = [num_emails, num_senders, num_respondents, skew_e, e_percentage, inactive_e, \
            e_nodes, e_edges, e_c_coef, e_mean_degree, e_long_tail, e_triangles, e_bidirected_edges]

    return social_net_features



with open('../monthly_features.csv', 'w') as f:

    things = ['active_devs','num_commits','num_committers','num_files','num_emails','num_senders','num_respondents',\
      'skew_c','skew_e','c_percentage','e_percentage','inactive_c','inactive_e',\
      'c_nodes','c_edges','c_c_coef','c_mean_degree','c_long_tail','c_triangles',\
      'e_nodes','e_edges','e_c_coef','e_mean_degree','e_long_tail','e_triangles', 'e_bidirected_edges']

    max_month = 110
    f.write('project_id,')
    for month in range(1, max_month +1):
        month = str(month)
        feature_names = [(month + '_' + t) for t in things]
        if month != max_month:
            f.write(','.join(feature_names) + ',')
        else:
            f.write(','.join(feature_names))
    f.write('\n')

for project in tqdm(projects):
    project_id = project.replace('.csv', '')
    # if it is a incubating project
    if int(project_id) < 49 or type(end_time_dict[project_id]) == float:
        continue
    # "10/16/2013"
    start_time = start_time_dict[project_id]
    end_time = end_time_dict[project_id]

    start_time_d = datetime.strptime(start_time, '%m/%d/%Y')
    end_time_d = datetime.strptime(end_time, '%m/%d/%Y')
    incubation_length = (end_time_d - start_time_d).days / 30

    commits = os.listdir(c_path + project)
    emails = os.listdir(e_path + project)

    # get all csvs
    csvs = commits + emails
    lengths = [int(c.replace('.csv', '')) for c in csvs]
    max_month = max(lengths)
    #max_length_t = max(max_length_t, max_month)

    for month in range(1, max_month+1):
        commit_path = c_path + project_id + os.sep + str(month) + '.csv'
        email_path = e_path + project_id + os.sep + str(month) + '.csv'

        # initiatilization
        if not os.path.exists(commit_path):
            inactive_c = 1
            num_commits = num_committers = num_files = skew_c = c_percentage = \
            c_nodes = c_edges = c_c_coef = c_mean_degree = c_long_tail = c_triangles = 0

        else:
            num_commits, num_committers, num_files, skew_c, c_percentage, inactive_c, \
            c_nodes, c_edges, c_c_coef, c_mean_degree, c_long_tail, c_triangles = cal_tehnical_net(commit_path)

        if not os.path.exists(email_path):
            inactive_e = 1
            num_emails = num_senders = num_respondents = skew_e = e_percentage = \
            e_nodes = e_edges = e_c_coef = e_mean_degree = e_long_tail = e_triangles = e_bidirected_edges = 0

        else:
            num_emails, num_senders, num_respondents, skew_e, e_percentage, inactive_e, \
            e_nodes, e_edges, e_c_coef, e_mean_degree, e_long_tail, e_triangles, e_bidirected_edges = cal_social_net(email_path)

        active_devs = num_committers + num_senders

        features = [active_devs,num_commits,num_committers,num_files,num_emails,num_senders,num_respondents, \
          skew_c,skew_e,c_percentage,e_percentage,inactive_c,inactive_e,\
          c_nodes,c_edges,c_c_coef,c_mean_degree,c_long_tail,c_triangles, \
          e_nodes,e_edges,e_c_coef,e_mean_degree,e_long_tail,e_triangles, e_bidirected_edges]

        features = [str(f) for f in features]

        with open('../monthly_features.csv', 'a') as f:
            if month == max_month:
                if month == 1:
                    f.write(project_id + ',' + ','.join(features))
                else:
                    f.write(','.join(features))
            elif month == 1:
                f.write(project_id + ',' + ','.join(features) + ',')
            else:
                f.write(','.join(features) + ',')

    with open('../monthly_features.csv', 'a') as f:
        f.write('\n')  

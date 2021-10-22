import json
import os
from tqdm import tqdm

commit_path = '../aggregated_data/commits/'
email_path = '../aggregated_data/emails/'

with open('./target_projects.txt', 'r') as f:
	projects = f.readlines()
projects = [p.replace('\n', '') for p in projects]


commit_dict = {}
email_dict = {}

num_emails = 0
num_commits = 0

for project in tqdm(projects):
	commit_dict[project] = {}
	if os.path.exists(commit_path + project + '.csv'):
		# counting commits stats
		with open(commit_path + project + '.csv', 'r') as f:
			commits = f.readlines()
		for line in commits:
			num_commits += 1
			time, committer, file = line.split(',')
			file = file.replace('\n', '')
			if committer not in commit_dict[project]:
				commit_dict[project][committer] = set()
			commit_dict[project][committer].add(file)

	email_dict[project] = {}

	if os.path.exists(email_path + project + '.csv'):
		# counting emails stats
		with open(email_path + project + '.csv', 'r') as f:
			emails = f.readlines()
		for line in emails:
			num_emails += 1
			time, sender, respondent  = line.split(',')
			respondent = respondent.replace('\n', '')
			if sender not in email_dict[project]:
				email_dict[project][sender] = set()
			email_dict[project][sender].add(respondent)

committers = set()
num_files = []

print('processing commits...')
for project in tqdm(commit_dict):
	committers = committers.union(set(commit_dict[project].keys()))
	file_list = commit_dict[project].values()
	file_set = set([item for sublist in file_list for item in sublist])
	num_files.append(len(file_set))

num_files = sum(num_files)


senders = set()
respondents = set()

print('processing emails...')
for project in tqdm(email_dict):
	senders = senders.union(set(email_dict[project].keys()))
	respondents_list = email_dict[project].values()
	respondents = respondents.union(set([item for sublist in respondents_list for item in sublist]))

communicators = senders.union(respondents)
contributors = committers.union(communicators)

both_commit_communicate = committers.intersection(communicators)

only_commit = len(committers) - len(both_commit_communicate)
only_communicate = len(communicators) - len(both_commit_communicate)
rest = len(contributors) - only_commit - only_communicate



print('We identify {} commits, through those, a total of {} files were modified.'.format(num_commits, num_files))

print('We identify {} unique contributors (by either committing or communicating), \
	   among them, {} only commit code, and {} only commucanite without committing code. \
	   the rest {} participated both activities.'\
	   .format(len(contributors), only_commit, only_communicate, rest))

print('We collect {} emails, from them we identify {} communicators. Among them, {} proactively participated discussions\
	activities (i.e., sending emails), the rest {} participated discussions in a relatively passive way (only received emails).'\
	.format(num_emails, len(communicators), len(senders), len(communicators)-len(senders)))











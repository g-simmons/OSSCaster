# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 21:02 
@author: Likang Yin
"""

from networkx.algorithms import community
from datetime import datetime
import networkx as nx
from math import floor
from math import ceil

#csv.field_size_limit(1000000000)

def get_edge_list_c(commits):

    # edges to be returned
    edges_list = []
    commit_dict = {}

    for commit in commits:
        commit_time, author, file = commit.replace('\n', '').split(',')
        
        if file not in commit_dict:
            commit_dict[file] = set()
        commit_dict[file].add(author)

    for file in commit_dict:
        commit_dict[file] = list(commit_dict[file])

        if len(commit_dict[file]) == 1:
            continue
        if len(commit_dict[file]) == 2:
            edges_list.append([commit_dict[file][0], commit_dict[file][1]])
            continue
        
        author_list = list(commit_dict[file])
        for i in range(len(author_list)-1):
            for j in range(i+1,len(author_list)):
                edges_list.append([commit_dict[file][i], commit_dict[file][j]])

    return edges_list

def get_edge_list_e(emails):
    # edges to be returned
    edges_list = []
    for email in emails:
        email_time, sender, respondent = email.replace('\n', '').split(',')
        edges_list.append([sender,respondent])

    return edges_list


def cal_both_sided_edges(edges_list):

    edge_dict = {}
    num = 0

    count_edge = {}

    for sender, corr in edges_list:
        if sender not in edge_dict:
            edge_dict[sender] = set()
        edge_dict[sender].add(corr)

    for sender, corr in edges_list:

        if sender in count_edge and corr in count_edge[sender]:
            continue

        if (corr in edge_dict) and (sender in edge_dict[corr]):
            num += 1
            if sender not in count_edge:
                count_edge[sender] = set()
            count_edge[sender].add(corr)

    return num

def get_network_features_e(emails):
    # get edges
    edges_list = get_edge_list_e(emails)
    G = nx.Graph()
    G.add_edges_from(edges_list)    #网络建立完成

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return [0, 0, 0, 0, 0]  #要返回边/节点比，均聚类系数，平均度

    a = num_edges/num_nodes
    avg_c = nx.average_clustering(G)

    num_both_sided_edges = cal_both_sided_edges(edges_list)

    nodes = G.nodes()
    degree_list = []
    for n in nodes:
        degree_list.append(G.degree(n))
    mean_degree = sum(degree_list) / num_nodes

    # return a,b,c
    return [num_nodes, num_edges, avg_c, mean_degree, num_both_sided_edges]


def get_communicability(path):

    edges_list = get_edge_list_e(path)
    G = nx.Graph()
    G.add_edges_from(edges_list)    #网络建立完成

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0 

    triangles_dict = nx.communicability(G)

    com_values = []
    for node in triangles_dict.keys():
        node = triangles_dict[node]
        com = list(node.values())
        com_values.append(sum(com)/len(com))

    communicability = sum(com_values)/len(com_values)

    print(communicability)

    return communicability

def get_triangles_c(commits):

    edges_list = get_edge_list_c(commits)
    G = nx.Graph()
    G.add_edges_from(edges_list)    #网络建立完成

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0 

    triangles_dict = nx.triangles(G)
    # When computing triangles for the entire graph each triangle is 
    # counted three times, once at each node. Self loops are ignored.
    num_tri = sum(triangles_dict.values()) / 3

    return num_tri


def get_triangles_e(emails):

    edges_list = get_edge_list_e(emails)
    G = nx.Graph()
    G.add_edges_from(edges_list)
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0 
    triangles_dict = nx.triangles(G)

    # When computing triangles for the entire graph each triangle is 
    # counted three times, once at each node. Self loops are ignored.
    num_tri = sum(triangles_dict.values()) / 3

    return num_tri

def get_network_features_c(commits):

    edges_list = get_edge_list_c(commits)
    G = nx.Graph()
    G.add_edges_from(edges_list)   

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0, 0, 0, 0

    a = num_edges/num_nodes
    avg_c = nx.average_clustering(G)

    nodes = G.nodes()
    degree_list = []
    for n in nodes:
        degree_list.append(G.degree(n))
    mean_degree = sum(degree_list) / num_nodes
    # return a,b,c
    return num_nodes, num_edges, avg_c, mean_degree

def get_interrupt_coe_email(emails):

    date_list = []

    for email in emails:
        email_time, sender, respondent = email.replace('\n', '').split(',')
        date_list.append(datetime.strptime(email_time, '%Y-%m-%d %H:%M:%S'))

    if len(date_list) == 0:
        return 1

    date_list.sort()
    duration = max((date_list[-1] - date_list[0]).days, 1)
    date_list2 = date_list[1:]

    # take the floor function to the days.
    # set the days < than 1 to 0
    diff = [max(floor((date_list2[i]-date_list[i]).days)-1,0) for i in range(len(date_list2))]
    diff.sort()

    # return top 3 in-active duration
    coe = sum(diff[-3:])/duration
    # coe = sum(diff[-3:])/duration
    return coe

def get_interrupt_coe_commit(commits):

    date_list = []
    for commit in commits:
        commit_time, author, file = commit.replace('\n', '').split(',')
        date_list.append(datetime.strptime(commit_time, '%Y-%m-%d %H:%M:%S'))

    if len(date_list) == 0:
        return 1

    date_list.sort()  
    duration = max((date_list[-1] - date_list[0]).days, 1)
    date_list2 = date_list[1:]
    diff = [max(floor((date_list2[i]-date_list[i]).days)-1,0) for i in range(len(date_list2))]
    diff.sort()
    # return top 3 gap
    coe = sum(diff[-3:])/duration
    return coe

def get_degree_longtail_e(emails):

    edges_list = get_edge_list_e(emails)
    G = nx.Graph()
    G.add_edges_from(edges_list)   

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0  
    nodes = G.nodes()
    degree_list = []
    for n in nodes:
        degree_list.append(G.degree(n))
    degree_list.sort(reverse=True)
    long_tail = (degree_list[0]-degree_list[-1])*0.25 + degree_list[-1]
    for i in range(len(degree_list)):
        if degree_list[i] <= long_tail:
            break
    l = i/num_nodes
    return i

def get_degree_longtail_c(commits):

    edges_list = get_edge_list_c(commits)
    G = nx.Graph()
    G.add_edges_from(edges_list)

    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    if num_edges == 0:
        return 0   

    nodes = G.nodes()
    degree_list = []

    for n in nodes:
        degree_list.append(G.degree(n))

    degree_list.sort(reverse=True)
    long_tail = (degree_list[0]-degree_list[-1])*0.25 + degree_list[-1]   # 四分之一位点

    for i in range(len(degree_list)):
        if degree_list[i] <= long_tail:
            break

    l = i/num_nodes

    return i




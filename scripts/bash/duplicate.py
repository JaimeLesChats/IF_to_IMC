import json
import pandas as pd

def remove():
    with open("dict.json","r") as d:
        dict = json.load(d)

    temp = dict.copy()
    for key in dict:
        if dict[key][0] == 1 :
            del temp[key]

    with open("dict.json","w") as d:
        json.dump(temp,d)

def filter():
    filtering_list = ['A','B','C','D','E','F','G']

    with open("dict.json","r") as d:
        dict = json.load(d)
    
    temp = dict.copy()
    for key in dict:
        if not all(elem in filtering_list for elem in dict[key][1:]):
            del temp[key]

    with open("dict.json","w") as d:
        json.dump(temp,d)

def create_csv():
    with open("dict.json","r") as d:
        dict = json.load(d)
    
    df = pd.DataFrame()
    cell_ids = []
    occurrences = []
    places = []

    for key in dict:
        cell_ids.append(int(key))
        occurrences.append(dict[key][0])
        places.append(dict[key][1])

    df["CellID"] = pd.Series(cell_ids)
    df["Occurrences"] = pd.Series(occurrences)
    df["Places"] = pd.Series(places)

    #df = df.sort_values(by='CellID')
    df = df.sort_values(by='Places')
    df.to_csv("./csv.csv",index = False)

    
    
remove()
filter()
create_csv()
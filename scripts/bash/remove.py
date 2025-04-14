import json

with open("dict.json","r") as d:
    dict = json.load(d)

temp = dict.copy()
for key in dict:
    if dict[key][0] == 1 :
        del temp[key]

with open("dict.json","w") as d:
            json.dump(temp,d)
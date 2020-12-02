import pandas as pd
import json
import string


def preprocess(sent):

    puncts = list(string.punctuation)
    puncts.remove('-')

    sent = [c for c in sent.lower() if c not in puncts or c == "'"]
    sent = ''.join([c for c in sent if not c.isdigit()])
    sent = sent.replace("person x", "personx").replace(" x's", " personx's").replace(" x ", " personx ")
    if sent[:2] == "x " or sent[:2] == "x'":
        sent = sent.replace("x ", "personx ", 1).replace("x'", "personx'")
    if sent[-3:] == " x\n":
        sent = sent.replace(" x\n", "personx\n")

    sent = sent.replace("person y", "persony").replace(" y's", " persony's").replace(" y ", " persony ")
    if sent[:2] == "y " or sent[:2] == "y'":
        sent = sent.replace("y ", "persony ", 1).replace("y'", "persony'")
    if sent[-3:] == " y\n":
        sent = sent.replace(" y\n", "persony\n")

    return sent.replace("personx", "Alex").replace("persony", "Jesse").lower()

df = pd.read_csv("v4_atomic_all.csv",index_col=0)
df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))


relations = []
relations += ["oEffect"]
relations += ["oReact"]
relations += ["oWant"]
relations += ["xAttr"]
relations += ["xEffect"]
relations += ["xIntent"]
relations += ["xNeed"]
relations += ["xReact"]
relations += ["xWant"]



atomic_data = {}
for event, row in df.iterrows():
    event = preprocess(event)
    if row['split'] not in atomic_data:
        atomic_data[row['split']] = {}
    for r in relations:
        for item in row[r]:
            try: atomic_data[row['split']]["{} {}".format(event,r)] += [item]
            except: atomic_data[row['split']]["{} {}".format(event,r)] = [item]



for split, enteries in atomic_data.items():
    for base_event, events in enteries.items():
        e = " [X_SEP] ".join(events) + " [X_SEP]"
        with open('data/atomic/{}.src'.format(split) , 'a+') as fp:
            fp.write(base_event + '\n')
        with open('data/atomic/{}.tgt'.format(split) , 'a+') as fp:
            fp.write(e + '\n')

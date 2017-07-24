import pandas as pd
from data.attribute_loader import _load_attributes

data = _load_attributes()

# data = pd.read_csv('fig1.csv')

table = {
    'intrans': ['', 'intransitive'],
    'trans_pers': ['', 'transitive for people'],
    'trans_obj': ['', 'transitive for objects'],
    'atomicity': ['Accomplishment', 'unclear without context',
                  'Activity', 'Achievement', 'State'],
    'energy': ['unclear without context', 'No motion', 'Low motion',
               'Medium motion', 'High motion'],
    'time': ['time n/a', 'On the order of seconds',
             'On the order of minutes', 'On the order of hours',
             'On the order of days'],
    'solitary': ['solitary','likely solitary','solitary or social','likely social','social'],
    'bodyparts_Arms': ['', 'uses arms'],
    'bodyparts_Head': ['', 'uses head'],
    'bodyparts_Legs': ['', 'uses legs'],
    'bodyparts_Torso': ['', 'uses torso'],
    'bodyparts_other': ['', 'uses a different bodypart'],
    'intrans_effect_0':  ['', 'the subject moves somewhere'],
    'intrans_effect_1': ['', 'the external world changes'],
    'intrans_effect_2': ['', 'the subject\'s state changes'],
   #'intrans_effect_3': ['', 'nothing changes'],
    'trans_obj_effect_0': ['', 'the object (a thing) moves somewhere'],
    'trans_obj_effect_1': ['', 'the external world changes'],
    'trans_obj_effect_2': ['', 'the object\'s state changes'],
   # 'trans_obj_effect_3': ['', 'nothing changes'],
    'trans_pers_effect_0': ['', 'the object (a person) moves somewhere'],
    'trans_pers_effect_1': ['', 'the external world changes'],
    'trans_pers_effect_2': ['', 'the object\'s state changes'],
  #  'trans_pers_effect_3': ['', 'nothing changes'],
}
for col in table:
    data[col] = [table[col][x] for x in data[col]]
data = data.set_index('template')
data.to_csv('fig1_processed.csv')

words = ['swig','drink','chug','slurp','sip','guzzle','gulp down','gurgle','drool','suck','swallow']
interesting = data.loc[words]
interesting.to_csv('fig1_interesting.csv')

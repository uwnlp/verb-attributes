# VerbsWithAttributes

This folder contains the Verbs With Attributes dataset, as well as some other goodies (namely, definitions for all of the verbs).

## attributes_split.csv

This is the data split that I used for my experiments. Although we collected annotations on the verb template level (consisting of a verb + an optional particle, ex. "put on", "put off", "put"), we split the dataset on the verb level.

The file `attributes_split.csv` has an additional column, `in_imsitu` which is True if that verb is in the imSitu database. To get the imSitu index, filter out all rows where in_imsitu is false, and then use the corresponding 0-index on the verbs. (If you don't want to do the filtering, you can also use the file `imsitu_verbs.txt`).

## verb_definitions.csv

This file contains definitions for our verb templates that I scraped from wordnik. I filtered out all definitions that weren't labeled as verbs, so you
can pretty much ignore the `POS` column. Note that some verb templates don't have definitions (so you'll have to backoff to verbs in those cases).

## attributes.csv

This contains all of the attributes we mined.

1. `intrans`: 1 if the verb can be used intransitively, 0 otherwise
1. `trans_pers`: 1 if the verb can be used in the form "<verb> someone", 0 otherwise
1. `intrans`: 1 if the verb can be used in the form "<verb> something", 0 otherwise
1. `atomicity`: Contains annotation of verbal aspect.
    - 0: Accomplishment
    - 1: unclear without context
    - 2: Activity
    - 3: Achievement
    - 4: State
1. `energy`: How much motion does the verb require
    - 0: unclear without context
    - 1: No motion
    - 2: Low motion
    - 3: Medium motion
    - 4: High motion
1. `solitary`: Whether the verb tends to be done in a solitary or social context
    - 0: solitary
    - 1: likely solitary
    - 2: solitary or social
    - 3: likely social
    - 4: social
1. `time`: how much time it takes to do that action
    - 0: N/A
    - 1: On the order of seconds
    - 2: On the order of minutes
    - 3: On the order of hours
    - 4: On the order of days
1. `bodyparts_Arms`: 1 if arms are used
1. `bodyparts_Head`: 1 if head is used
1. `bodyparts_Legs`: 1 if legs are used
1. `bodyparts_Torso`: 1 if torso is used
1. `bodyparts_other`: 1 if another body part is used
1. `emot_anger`: 1 if verb is highly associated with anger
1. `emot_another`: 1 if verb is highly associated with another emotion
1. `emot_fear`: 1 if verb is highly associated with fear
1. `emot_happy`: 1 if verb is highly associated with happiness
1. `emot_sad`: 1 if verb is highly associated with sadness
1. `emot_surp`: 1 if verb is highly associated with surprisal
1. `intrans_effect_0`: 1 if the subject moves somewhere
1. `intrans_effect_1`: 1 if the external world changes
1. `intrans_effect_2`: 1 if the subject's state changes
1. `intrans_effect_3`: 1 if nothing changes
1. `trans_obj_effect_0`: 1 if the object (a thing) moves somewhere
1. `trans_obj_effect_1`: 1 if the external world changes
1. `trans_obj_effect_2`: 1 if the object's state changes
1. `trans_obj_effect_3`: 1 if nothing changes
1. `trans_pers_effect_0`: 1 if the object (a person) moves somewhere
1. `trans_pers_effect_1`: 1 if the external world changes
1. `trans_pers_effect_2`: 1 if the object's state changes
1. `trans_pers_effect_3`: 1 if nothing changes

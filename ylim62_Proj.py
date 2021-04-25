import pandas as pd
import numpy as np

# 1. read data
ltable = pd.read_csv("data/ltable.csv")
rtable = pd.read_csv("data/rtable.csv")
train = pd.read_csv("data/train.csv")

# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id

    # get array version of the candset
    pairs = np.array(candset)

    # get the correct ids from the pairs ([:, 0/1]), then get all
    # columns for that id (so use : instead of "price" for example)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]

    # rename the columns
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]

    # reset index in place (no new obj) and drop index without
    # adding it as a column to new df
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)

    # concat them
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

def block_data(ltable, rtable):
    # block by brand
    candset_brd = block_by_attr(ltable, rtable, "brand")
    print("UPDATE: Finished blocking by brand (primary)\n")

    # block by modelno
    candset_mdl = block_by_attr(ltable, rtable, "modelno")
    print("UPDATE: Finished blocking by modelno (secondary)\n")

    # get the union between the two sets
    candset_brd_set = set(map(tuple, candset_brd))
    candset_mdl_set = set(map(tuple, candset_mdl))
    candset_tot_set = candset_brd_set.union(candset_mdl_set)
    candset_tot = list(map(list, candset_tot_set))
    print("UPDATE: Finished finding union\n")

    return candset_tot

def block_by_attr(ltable, rtable, attr):
    # check if attr is brand
    is_brand = attr == "brand"

    # ensure attr is str
    ltable[attr] = ltable[attr].fillna("").astype(str)
    rtable[attr] = rtable[attr].fillna("").astype(str)

    # get all attr
    attrs_l = set(ltable[attr].values)
    attrs_r = set(rtable[attr].values)
    #brands = brands_l.union(brands_r)

    # map each attr to left ids and right ids
    attr2ids_l = {a.lower(): [] for a in attrs_l}
    attr2ids_r = {a.lower(): [] for a in attrs_r}
    for i, x in ltable.iterrows():
        attr2ids_l[x[attr].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        attr2ids_r[x[attr].lower()].append(x["id"])

    # map each attr to left titles and right titles
    attr2ttls_l = {a.lower(): [] for a in attrs_l}
    attr2ttls_r = {a.lower(): [] for a in attrs_r}
    for i, x in ltable.iterrows():
        attr2ttls_l[x[attr].lower()].append(x["title"])
    for i, x in rtable.iterrows():
        attr2ttls_r[x[attr].lower()].append(x["title"])

    # put id pairs that share the same attr in candidate set
    candset_attr = []
    for attr_l in attrs_l:
        for attr_r in attrs_r:
            # make easier for comparisons: no spaces, and no non-alphanumerics
            attr_l_clean = ''.join(c for c in attr_l.replace(" ", "") if c.isalnum())
            attr_r_clean = ''.join(c for c in attr_r.replace(" ", "") if c.isalnum())

            alc_empty = attr_l_clean == ""
            arc_empty = attr_r_clean == ""

            # get the ids
            l_ids = attr2ids_l[attr_l.lower()]
            r_ids = attr2ids_r[attr_r.lower()]

            # get the titles
            l_ttls = attr2ttls_l[attr_l.lower()]
            r_ttls = attr2ttls_r[attr_r.lower()]

            # if one of the attr are empty (only if brand, since that
            # is the primary blocking attr) or if the (non-empty) attr (no
            # spaces) are equal or if one is a substring of the other
            if ((attr_l == "" and is_brand) or
                (attr_r == "" and is_brand) or
                (attr_l_clean == attr_r_clean and not alc_empty and not arc_empty) or
                (attr_l_clean in attr_r_clean and not alc_empty) or
                (attr_r_clean in attr_l_clean and not arc_empty)):

                for id_l in l_ids:
                    for id_r in r_ids:
                        candset_attr.append([id_l, id_r])
            else:
                for i, ttl_l in enumerate(l_ttls):
                    # make easier for comparisons: no spaces, and no non-alphanumerics
                    ttl_l_clean = ttl_l.replace(" ", "")
                    ttl_l_clean = ''.join(c for c in ttl_l_clean if c.isalnum())

                    # if attr can be found in title (only if attr is not empty)
                    if (attr_r_clean in ttl_l_clean and not arc_empty):
                        for id_r in r_ids:
                            candset_attr.append([l_ids[i], id_r])

                for i, ttl_r in enumerate(r_ttls):
                    # make easier for comparisons: no spaces, and no non-alphanumerics
                    ttl_r_clean = ttl_r.replace(" ", "")
                    ttl_r_clean = ''.join(c for c in ttl_r_clean if c.isalnum())

                    # if attr can be found in title (only if attr is not empty)
                    if (attr_l_clean in ttl_r_clean and not alc_empty):
                        for id_l in l_ids:
                            candset_attr.append([id_l, r_ids[i]])

    return candset_attr

# blocking to reduce the number of pairs to be compared
candset = block_data(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking", len(candset))
print()

candset_df = pairs2LR(ltable, rtable, candset)
print("UPDATE: Finished making Dataframe from candset\n")

# 3. Feature engineering

import textdistance as td

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())

    return td.jaccard(x, y)

def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()

    return td.levenshtein.normalized_similarity(x, y)

def ratcliff_obershelp_similarity(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()

    return td.ratcliff_obershelp(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []

    # creates array 10 rows, alterating between j_sim and l_dist of a
    # given attr (ex. [j for title], [l for title], [j for cat], [l for cat]...)
    for attr in attrs:
        # have 3 different features: token-based (j_sim), edit-based (l_dist), sequence-based (ro_sim)
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        ro_sim = LR.apply(ratcliff_obershelp_similarity, attr=attr, axis=1)
        
        # add it to the features
        features.append(j_sim)
        features.append(l_dist)
        features.append(ro_sim)
    
    # flips array so that it is # of pairs many rows, each with a single
    # pair's j and l scores for all of their attrs
    features = np.array(features).T
    return features

candset_features = feature_engineering(candset_df)
print("UPDATE: Finished feature engineering on candset\n")

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values
print("UPDATE: Finished feature engineering on training set\n")

# 4. Model training and prediction

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_label)
y_pred_rf = rf.predict(candset_features)
#y_pred = rf.predict(candset_features)
print("UPDATE: Finished model training and predicting with Random Forest\n")

gnb = GaussianNB()
gnb.fit(training_features, training_label)
y_pred_gnb = gnb.predict(candset_features)
print("UPDATE: Finished model training and predicting with Gaussian NB\n")

kn = KNeighborsClassifier()
kn.fit(training_features, training_label)
y_pred_kn = kn.predict(candset_features)
print("UPDATE: Finished model training and predicting with K-Nearest Neighbors\n")

y_pred = []
for i in range(len(candset_features)):
    y_avg = (y_pred_rf[i] + y_pred_gnb[i] + y_pred_kn[i])/3
    
    if (y_avg > 0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred = np.array(y_pred)
print("UPDATE: Finished finding concensus among predictions\n")

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)
print("UPDATE: Finished creating output file\n")
import pandas as pd #importing pandas library
#from sklearn.neural_network import MLPClassifier
matches = pd.read_csv("premierleague.csv", index_col=0) #reading in match data
matches["date"] = pd.to_datetime(matches["date"])#c
matches["venue_code"] = matches["venue"].astype("category").cat.codes #changes home and away to 0 and 1 repsectivly
matches["opp_code"] = matches["opponent"].astype("category").cat.codes #assigns each possible opponent a number
matches["hour"] = matches["time"].str.replace(":.+","",regex=True).astype("int")
matches["expected goals"] = matches["xg"].round(0)
matches["day_code"] = matches["date"].dt.dayofweek # assigns each day of the week a number 6 being sunday 
#matches["target"] = matches["result"].astype("category").cat.codes#target changes to 0,1,2 whether team draws,losses or wins respectively
matches["target"] = (matches["result"] =="W").astype("int")
from sklearn.ensemble import RandomForestClassifier #importing machine learning
rf = RandomForestClassifier(max_depth=20, min_samples_leaf=8, min_samples_split=2, n_estimators=200, random_state=1 )
#rf = MLPClassifier(random_state=1, activation='tanh', alpha=0.01, hidden_layer_sizes=(100,))
#rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1 ) #making parameters 
train = matches[matches["date"] < '2021-08-12'] #spliting data in to training set
test = matches[matches["date"] > '2021-08-12']#and test set 
predictors = ["venue_code","opp_code","hour","day_code","poss","season","expected goals"]#creating our predictors
rf.fit(train[predictors], train["target"])#fit our randomforest model
preds = rf.predict(test[predictors])#generate predictions
from sklearn.metrics import accuracy_score#this is how we will determine the accuracy of the model
acc = accuracy_score(test["target"], preds)#testing the accuracy
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))#seeing how how accurate each result is 
pd.crosstab(index=combined["actual"], columns=combined["prediction"])  
from sklearn.metrics import precision_score
precision_score(test["target"], preds)
grouped_matches = matches.groupby("team")#creates a dataframe for every squad
group = grouped_matches.get_group("Manchester City")#all city games
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()#will only use averages before the matchweek rather than using results during which doesnt make sense when making future predictions
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)#removes rows with missing values
    return group
cols = ["gf","ga","sh","sot","dist","fk","pk","pkatt","poss","xg"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group, cols, new_cols)
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols,new_cols))#we made rolling averages for every team and their matches
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])
def make_predictions(data, predictors):
    train = data[data["date"] < '2021-08-12']
    test = data[data["date"] > '2021-08-12']
    rf.fit(train[predictors], train["target"])#fit our randomforest model
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
#precision 
combined = combined.merge(matches_rolling[["date","team","opponent","result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd", 
    "Newcastle United": "Newcastle Utd", 
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
combined


merged = combined.merge(combined, left_on=["date","new_team"], right_on=["date","opponent"])
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()
#merged.to_csv("predictions.csv")
merged
print(acc)
#matches.columns
print(precision)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'min_samples_split': [2, 5, 10, 20, 40],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 8]
}
rf = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(train[predictors], train["target"])
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
preds = best_model.predict(test[predictors])
acc = accuracy_score(test["target"], preds)
print("Accuracy of best model:", acc)

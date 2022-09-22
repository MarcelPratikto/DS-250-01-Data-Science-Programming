#===================================================================================================
# Imports
#===================================================================================================
import pandas as pd
import altair as alt
import numpy as np
import sqlite3

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#===================================================================================================
# Read files
#===================================================================================================
url = "filename.csv"

new_column_names = ["column1","column2"]

data = pd.read_csv(url, encoding="ISO-8859-1", header=None, skiprows=2)
data.columns = new_column_names

con = sqlite3.connect('lahmansbaseballdb.sqlite')
teams = pd.read_sql_query("""
SELECT teamid, AVG(W), AVG(L), AVG(G) as avg_game, round((AVG(W)+0.0)/AVG(G)*100, 3) as avgwin_game
FROM teams
GROUP BY teamid
HAVING avg_game >= 100
ORDER BY avgwin_game DESC
LIMIT 5
""", con)
byuiplayers = pd.read_sql_query("""
    SELECT p.playerid, s.teamid, s.salary, s.yearid, c.schoolid
    FROM people AS p
        JOIN salaries AS s ON p.playerid = s.playerid
        JOIN collegeplaying as c ON p.playerid = c.playerid
    WHERE c.schoolid = 'idbyuid'
    ORDER BY salary DESC
""", con)
df = pd.read_sql_query("SELECT * FROM fielding LIMIT 5", con)

#===================================================================================================
# Print data
#===================================================================================================
# table
print(data.head().to_markdown(index=False, tablefmt="grid"))

# boxplot
gartype_chart = (alt.Chart(denver)
    .mark_boxplot(
        size = 50
    )
    .properties(
        width = 900
    )
    .encode(
        x = alt.X('gartype'),
        y = alt.Y('yrbuilt',
                  scale = alt.Scale(zero = False),
                  axis = alt.Axis(format='d'))
    )
)

# practice plot with circles, lines, text
graph1 = alt.Chart(clean, title="This is Awesome").mark_circle(color="red").encode(
    x = alt.X("hp", axis = alt.Axis(title="Horse Power")),
    y = alt.Y("mpg", axis = alt.Axis(title = "Miles per Gallon"))
)
graph1
junk_df1 = pd.DataFrame({"hp": [80]})
junk_df2 = pd.DataFrame({"hp": [160]})
line1 = (alt.Chart(junk_df1).mark_rule(strokeDash=[4,4] , color="red").encode(
    x="hp")
)
line2 = (alt.Chart(junk_df2).mark_rule(strokeDash=[4,4] , color="red").encode(
    x="hp")
)
final_graph = graph1 + line1 + line2
final_graph

# bar plot
visual_2 = (alt.Chart(favorite_df)
    .mark_bar()
    .encode(
        x = alt.X("percent_fav", axis=alt.Axis(title="Percent Favorite")),
        y = alt.Y("name", axis=alt.Axis(title="Movie Name"))        
    )
    .properties(title="What's the best Star Wars Movie?")
)
visual_2
#visual_2.save("visual_2.png")

#===================================================================================================
# Filter data
#===================================================================================================
data = data[data["column1"] == "Yes"]
data = data[data["column1"] != np.nan]

dat3 = dat2.dropna(subset=['car'])

# income category
incomes = []
for income in sw_data["Household Income"]:
    if income not in incomes:
        incomes.append(income)
incomes
# one-hot encode
# Create an additional column that converts the income ranges to a number and drop the income range categorical column.
new_income = []
for income in sw_data["Household Income"]:
    if income == "$0 - $24,999":
        income = 1
    elif income == "$25,000 - $49,999":
        income = 2
    elif income == "$50,000 - $99,999":
        income = 3
    elif income == "$100,000 - $149,999":
        income = 4
    elif income == "$150,000+":
        income = 5
    else:
        income = 0
    new_income.append(income)
sw_data["Household Income"] = new_income

# fill in na values
dat2 = dat.copy()
dat2.hp = dat2.hp.fillna(value = hp_mean)

# groupby, filter
flights_mean = (flights
    .assign(minutes_per_total_delays = lambda x: x.minutes_delayed_total / x.num_of_delays_total)
    .assign(hours_per_total_delays = lambda x: x.minutes_per_total_delays / 60)
    .groupby(["airport_code"]).mean()
    .filter(["airport_code", "airport_name", "minutes_delayed_total", "num_of_delays_total", "hours_per_total_delays"])
    .sort_values(by="hours_per_total_delays", ascending=False)
    .reset_index()
)

christian_names = names.query("name in ['Mary', 'Martha', 'Paul', 'Peter'] & year >= 1920 & year <= 2000")


#===================================================================================================
# Machine Learning
#===================================================================================================
# see the names of all that columns
data.columns
# pick columns that are relevant to the machine learning model
x = data.filter(["is_fan", "fan_star_trek", "Gender", "Age", "Education"])
# choose what to compare it to (target)
y = data["over_50k"]
# make sure the number of output matches
print(data.shape)
print(x.shape)
print(y.shape)
# create the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.081, random_state = 864)
classifier = RandomForestClassifier()
# train the model
classifier.fit(x_train, y_train)
# make predictions
y_predictions = classifier.predict(x_test)
# test how accurate predictions are
metrics.accuracy_score(y_test, y_predictions)

# Confusion matrix
print(metrics.classification_report(y_test, y_predictions))
predictions = classifier.predict(x_test)
con_matrix = confusion_matrix(y_test, predictions)
plot_confusion_matrix(classifier, x_test, y_test, cmap = 'Blues')
predictions
y_test
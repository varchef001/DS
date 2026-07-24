import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


print("ZOMATO DATA ANALYSIS PROJECT")
print("=" * 60)


try:
    df = pd.read_csv("zomato.csv")
except UnicodeDecodeError:
    df = pd.read_csv("zomato.csv", encoding="latin1")


print("\nSTEP 1: FIRST 5 ROWS")
print("-" * 60)
print(df.head())


print("\nSTEP 2: DATASET SHAPE")
print("-" * 60)
print(df.shape)

print("\nCOLUMN NAMES")
print("-" * 60)
print(df.columns)

print("\nDATA TYPES")
print("-" * 60)
print(df.dtypes)


print("\nSTEP 3: MISSING VALUES BEFORE CLEANING")
print("-" * 60)
print(df.isnull().sum())


df = df.rename(columns={
    "approx_cost(for two people)": "cost_for_two",
    "listed_in(type)": "service_type",
    "listed_in(city)": "listed_city"
})


df = df.drop(columns=[
    "url",
    "address",
    "phone",
    "reviews_list",
    "menu_item"
], errors="ignore")


df["rate"] = df["rate"].astype(str)

df["rate"] = df["rate"].replace(["NEW", "-", "nan"], np.nan)

df["rate"] = df["rate"].str.replace("/5", "", regex=False)

df["rate"] = df["rate"].str.strip()

df["rate"] = pd.to_numeric(df["rate"], errors="coerce")


df["cost_for_two"] = df["cost_for_two"].astype(str)

df["cost_for_two"] = df["cost_for_two"].str.replace(",", "", regex=False)

df["cost_for_two"] = df["cost_for_two"].replace("nan", np.nan)

df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce")


df["rate"] = df["rate"].fillna(df["rate"].median())

df["cost_for_two"] = df["cost_for_two"].fillna(df["cost_for_two"].median())

df["votes"] = df["votes"].fillna(0)

text_columns = [
    "name",
    "online_order",
    "book_table",
    "location",
    "rest_type",
    "dish_liked",
    "cuisines",
    "service_type",
    "listed_city"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")


print("\nSTEP 9: SHAPE BEFORE REMOVING DUPLICATES")
print("-" * 60)
print(df.shape)

df = df.drop_duplicates()

print("\nSHAPE AFTER REMOVING DUPLICATES")
print("-" * 60)
print(df.shape)


print("\nSTEP 10: MISSING VALUES AFTER CLEANING")
print("-" * 60)
print(df.isnull().sum())

print("\nFINAL DATASET INFO")
print("-" * 60)
df.info()


print("\nSTEP 11: ONLINE ORDER COUNT")
print("-" * 60)

online_order_count = df["online_order"].value_counts()

print(online_order_count)


print("\nSTEP 12: TABLE BOOKING COUNT")
print("-" * 60)

table_booking_count = df["book_table"].value_counts()

print(table_booking_count)


print("\nSTEP 13: TOP 10 CUISINES")
print("-" * 60)

top_cuisines = df["cuisines"].value_counts().head(10)

print(top_cuisines)


print("\nSTEP 14: TOP 10 LOCATIONS")
print("-" * 60)

top_locations = df["location"].value_counts().head(10)

print(top_locations)


print("\nSTEP 15: RATING SUMMARY")
print("-" * 60)

print(df["rate"].describe())


print("\nSTEP 16: COST SUMMARY")
print("-" * 60)

print(df["cost_for_two"].describe())


print("\nSTEP 17: COST VS RATING CORRELATION")
print("-" * 60)

cost_rating_correlation = df["cost_for_two"].corr(df["rate"])

print(cost_rating_correlation)


print("\nSTEP 18: ONLINE ORDER VS RATING")
print("-" * 60)

online_rating = df.groupby("online_order")["rate"].mean()

print(online_rating)


print("\nSTEP 19: TABLE BOOKING VS RATING")
print("-" * 60)

booking_rating = df.groupby("book_table")["rate"].mean()

print(booking_rating)


print("\nSTEP 20: ONLINE ORDER VS COST")
print("-" * 60)

online_cost = df.groupby("online_order")["cost_for_two"].mean()

print(online_cost)


print("\nSTEP 21: TABLE BOOKING VS COST")
print("-" * 60)

booking_cost = df.groupby("book_table")["cost_for_two"].mean()

print(booking_cost)


top_locations.plot(kind="bar")

plt.xlabel("Location")
plt.ylabel("Number of Restaurants")
plt.title("Top 10 Restaurant Locations")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


plt.figure(figsize=(6, 6))

online_order_count.plot(kind="pie", autopct="%1.1f%%")

plt.ylabel("")
plt.title("Online Order Availability")
plt.tight_layout()

plt.show()


plt.figure(figsize=(8, 5))

plt.hist(df["rate"], bins=20)

plt.xlabel("Rating")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Restaurant Ratings")
plt.tight_layout()

plt.show()


plt.figure(figsize=(7, 5))

sns.boxplot(x="online_order", y="rate", data=df)

plt.xlabel("Online Order")
plt.ylabel("Rating")
plt.title("Rating vs Online Order")
plt.tight_layout()

plt.show()


plt.figure(figsize=(8, 5))

plt.scatter(df["cost_for_two"], df["rate"], alpha=0.4)

plt.xlabel("Cost for Two")
plt.ylabel("Rating")
plt.title("Cost vs Rating")
plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 5))

top_cuisines.plot(kind="bar")

plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.title("Top 10 Cuisines")
plt.xticks(rotation=75)
plt.tight_layout()

plt.show()


print("\nSTEP 28: CORRELATION ANALYSIS")
print("-" * 60)

numeric_data = df[["rate", "votes", "cost_for_two"]]

correlation_table = numeric_data.corr()

print(correlation_table)


plt.figure(figsize=(6, 4))

sns.heatmap(correlation_table, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.show()


sample_data = numeric_data.sample(1000, random_state=42)

sns.pairplot(sample_data)

plt.show()


print("\nSTEP 31: RATING PREDICTION MODEL")
print("-" * 60)

model_data = df[["votes", "cost_for_two", "rate"]].copy()

X = model_data[["votes", "cost_for_two"]]

y = model_data["rate"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nR2 Score:")
print(r2)


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(feature_importance)


print("\nSTEP 37: INSIGHTS")
print("=" * 60)

highest_restaurant_location = df["location"].value_counts().idxmax()

most_common_cuisine = df["cuisines"].value_counts().idxmax()

average_rating = df["rate"].mean()

average_cost = df["cost_for_two"].mean()


print("\nInsight 1:")
print("The location with the most restaurants is", highest_restaurant_location)

print("\nInsight 2:")
print("The most common cuisine is", most_common_cuisine)

print("\nInsight 3:")
print("The average restaurant rating is", round(average_rating, 2))

print("\nInsight 4:")
print("The average cost for two people is", round(average_cost, 2))

print("\nInsight 5:")
print("Average rating based on online order:")
print(online_rating)

print("\nInsight 6:")
print("Average rating based on table booking:")
print(booking_rating)


print("\nSTEP 38: BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("\nRecommendation 1:")
print("Restaurants should offer online ordering because it gives customers more convenience.")

print("\nRecommendation 2:")
print("New restaurants can focus on popular locations where restaurant demand is high.")

print("\nRecommendation 3:")
print("Restaurants should focus on popular cuisines because they already have strong customer interest.")

print("\nRecommendation 4:")
print("Restaurants can improve ratings by improving food quality, service quality, and customer experience.")

print("\nRecommendation 5:")
print("Table booking can be useful for restaurants that want to attract dine-in customers.")


print("\nSTEP 39: CONCLUSION")
print("=" * 60)

print("""
In this project, we analyzed the Zomato restaurant dataset.

First, we loaded and understood the data.
Next, we cleaned the rating and cost columns.
Then, we handled missing values and removed duplicates.
After that, we performed exploratory data analysis.
We also created visualizations such as bar charts, pie charts, histograms, box plots, scatter plots, and heatmaps.
Finally, we created a simple Linear Regression model to predict restaurant ratings.

The analysis showed important patterns in online ordering, table booking, locations, cuisines, ratings, and cost.
""")
# ============================================
# SMART CANTEEN DEMAND PREDICTION
# MongoDB + ML + Graphs
# ============================================

print("Starting Program...\n")

# ---------- IMPORTS ----------
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================
# 1️⃣ CONNECT TO MONGODB
# ============================================

client = MongoClient("mongodb://localhost:27017/")
db = client["smart_canteen_ml"]        # YOUR DATABASE NAME
collection = db["canteen_sales"]      # YOUR COLLECTION NAME

print("Connected to MongoDB!")

# ============================================
# 2️⃣ FETCH DATA
# ============================================

data = list(collection.find())

if len(data) == 0:
    print("No data found in MongoDB collection!")
    exit()

# Remove _id column
for record in data:
    record.pop("_id", None)

df = pd.DataFrame(data)

print("Data fetched successfully!")
print("Total Records:", len(df))
print(df.head(), "\n")

# ============================================
# 3️⃣ ENCODE CATEGORICAL COLUMNS
# ============================================

le_day = LabelEncoder()
le_weather = LabelEncoder()
le_exam = LabelEncoder()

df["day_of_week"] = le_day.fit_transform(df["day_of_week"])
df["weather"] = le_weather.fit_transform(df["weather"])
df["exam_period"] = le_exam.fit_transform(df["exam_period"])

# ============================================
# 4️⃣ FEATURES & TARGET
# ============================================

X = df.drop("today_sales", axis=1)
y = df["today_sales"]

# ============================================
# 5️⃣ TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 6️⃣ TRAIN MODEL
# ============================================

model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
# 7️⃣ PREDICTION
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 8️⃣ EVALUATION
# ============================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MODEL PERFORMANCE")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# ============================================
# 📊 GRAPHS
# ============================================

# 1️⃣ Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# 2️⃣ Sales Distribution
plt.figure()
sns.histplot(df["today_sales"], kde=True)
plt.title("Distribution of Today Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# 3️⃣ Previous Sales vs Today Sales
plt.figure()
plt.scatter(df["previous_sales"], df["today_sales"])
plt.xlabel("Previous Sales")
plt.ylabel("Today Sales")
plt.title("Previous Sales vs Today Sales")
plt.show()

print("\nProgram Finished Successfully!")
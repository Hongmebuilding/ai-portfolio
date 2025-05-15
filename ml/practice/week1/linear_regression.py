from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
data = fetch_california_housing()
X, y = data.data, data.target

# 학습용/테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 선언 + 학습
model = LinearRegression()
model.fit(X_train, y_train) # x_train으로 y_train을 가장 잘 예측하는 직선 또는 평면을 찾는 과정

# 예측
pred = model.predict(X_test)

# 시각화
plt.scatter(y_test, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression")
plt.show()

# 성능 평가
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("평균 제곱 오차 (MSE):", mse)
print("결정 계수 (R² score):", r2)
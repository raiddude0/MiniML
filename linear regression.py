import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('synthetic_studytime_examscore.csv')

data = data.rename(columns={'study_time_hours_per_week': 'study_time','exam_score' : 'score'})
data = data.drop(columns=['student_id'])

def loss_function(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i]['study_time']
        y = points.iloc[i]['score']
        predicted_y = m * x + b
        total_loss += (predicted_y - y) ** 2
    return total_loss / len(points)

def gradient_descent(m, b, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = len(points)
    
    for i in range(N):
        x = points.iloc[i]['study_time']
        y = points.iloc[i]['score']
        predicted_y = m * x + b
        m_gradient += -(2/N) * x * (y - predicted_y)
        b_gradient += -(2/N) * (y - predicted_y)
    
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient
    
    return m, b

m = 0
b = 0
learning_rate = 0.0001
epochs = 1000
for epoch in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)
print(f"Final parameters: m = {m}, b = {b}")
plt.scatter(data['study_time'], data['score'], color='blue', label='Data Points')
x_values = [data['study_time'].min(), data['study_time'].max()] 
y_values = [m * x + b for x in x_values]
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.xlabel('Study Time (hours/week)')
plt.ylabel('Exam Score')
plt.title('Linear Regression')
plt.legend()
plt.show()

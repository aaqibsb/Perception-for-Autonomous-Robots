import cv2 as cv
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

init_printing(use_unicode=False, wrap_line=False)

x = []
y = []

""" Reading video """
vid = cv.VideoCapture('Videos/ball.mov')
while True:
    # Read in video
    isTrue, frame = vid.read()

    # Convert to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Set Threshold Values to convert red to white pixels
    lower_color = np.array([0,166,50], dtype=np.uint8)
    upper_color = np.array([255,255,255], dtype=np.uint8)

    # Create a mask
    mask = cv.inRange(hsv, lower_color, upper_color)

    # Using Median Blur and Erode function to remove outliers
    median = cv.medianBlur(mask, 17)
    eroded = cv.erode(median, (21, 21), iterations=10)

    # Find coordinates of all white pixels
    coord = cv.findNonZero(eroded)

    # Loop to remove certain outliers and bypass errors when ball is not detected
    if coord is not None:
        if coord[0,0,0] != 1217 and coord[0,0,0] != 0:
            a = coord.mean(axis=0)

            # Appending x and y coordinates for plotting
            x.append(int(a[0,0]))
            y.append(int(a[0,1])-5)

    # Loop to plot centre point and circle while the video is playing
    if len(x) > 0:
        for i in range(0,len(x) - 1):
            cv.line(frame, (x[i],y[i]), (x[i+1],y[i+1]), (255, 255, 255), 2)
            cv.circle(frame, (x[i],y[i]), 10, (0, 0, 255), 1)

    cv.imshow('Frame', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

vid.release()
cv.destroyAllWindows()

# Plotting the detected trajectory
plt.plot(x,y)

""" Least Square Method to estimate a parabolic curve """
XY = []
X2Y = []
X2 = []
X3 = []
X4 = []
for i in range(0,len(x)):
    XY.append(x[i]*y[i])
    X2Y.append((x[i]**2)*y[i])
    X2.append(x[i]**2)
    X3.append(x[i]**3)
    X4.append(x[i]**4)

# Finding sum
X_s = sum(x)
Y_s = sum(y)
XY_s = sum(XY)
X2Y_s = sum(X2Y)
X2_s = sum(X2)
X3_s = sum(X3)
X4_s = sum(X4)

# Creating symbols to formulate equation
a, b, c, x_sym, y_sym, x2_sym = symbols('a b c X Y X\u00B2')

# Solving for coefficients
res = solve([Eq(((len(x)*a) + (X_s*b) + (X2_s*c)), Y_s), Eq(((X_s*a) + (X2_s*b) + (X3_s*c)), XY_s),
      Eq(((X2_s*a) + (X3_s*b) + (X4_s*c)), X2Y_s)], [a, b, c])


print("Equation of Trajectory:")
pprint(simplify(Eq((res[a]) + (res[b])*x_sym + (res[c])*x2_sym,y_sym)))


# Appending Y coordinates for the estimated curve
Y = []
for i in range(0,len(x)):
    Y.append(simplify(res[a] + res[b]*x[i] + res[c]*x[i]*x[i]))

plt.plot(x,y, label='Trajectory of Ball')
plt.plot(x,Y, label='Estimated Trajectory using Std. Least Squares')

plt.title('Ball Tracking')
plt.xlabel('X')
plt.ylabel('-Y')

plt.legend()
plt.grid()
plt.show()

# Calculating the x-coordinate of Landing Spot
LandingSpot = Eq(res[a] + (res[b])*x_sym + (res[c])*x2_sym - y_sym,0)
LS_eq = LandingSpot.subs(x2_sym, x_sym*x_sym).subs(y_sym, y[0]+300)
LS_res = solve(LS_eq,x_sym)

print()
print("Landing Spot: ", round(LS_res[1].evalf()))



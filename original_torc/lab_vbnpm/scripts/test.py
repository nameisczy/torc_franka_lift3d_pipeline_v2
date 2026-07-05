from scipy.interpolate import CubicHermiteSpline

import rospy
from utils.conversions import float_to_ros_duration


print("Spline 1:")

x = [0.0, 1.0, 2.0, 3.0]
y = [2.0, -1.0, 1.5, 2.3]
dydx = [-2.0, 2.5, 0.8, 0]

spline = CubicHermiteSpline(x=x, y=y, dydx=dydx)
print(spline)


try:
    print("Spline 2 (ValueError: `x` must be strictly increasing sequence.):")
    x = [0.0, 1.0, 1.0, 3.0]
    y = [2.0, -1.0, 1.5, 2.3]
    dydx = [-2.0, 2.5, 0.8, 0]

    spline = CubicHermiteSpline(x=x, y=y, dydx=dydx)
    print(spline)
except Exception as e:
    print("ERROR: ", e)


try:
    print("Spline 3 (ValueError: `x` must contain at least 2 elements.):")
    x = [0.0]
    y = [2.0]
    dydx = [-2.0]

    spline = CubicHermiteSpline(x=x, y=y, dydx=dydx)
    print(spline)
except Exception as e:
    print("ERROR: ", e)


print("rospy duration: ", rospy.Duration(1.2))

for i in range(50):
    value = i / 1e9
    print(
        "value: ",
        str(value).ljust(8),
        " float_to_ros_duration: ",
        float_to_ros_duration(value, 9),
    )


# for i in range(1000):
#     print(1 + i / 1000, float_to_ros_duration(1 + i / 1000, 13))

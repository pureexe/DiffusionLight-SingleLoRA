import math 

theta = math.radians(7.5)
inv_theta = math.radians(90-7.5)

d = math.sin(inv_theta) / math.tan(theta) + math.cos(inv_theta)
print(d)


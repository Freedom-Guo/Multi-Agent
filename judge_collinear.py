# method 1 using vector's relationship:判断向量(p1-->p2)和向量(p1-->p3)的斜率是否相等

def func1(p1, p2, p3):
    return abs((p3[1]-p1[1])*(p2[0]-p1[0]) - (p2[1]-p1[1]) * (p3[0]-p1[0])) < 1e-6
    
# method 2 using matrix, 用行列式求三角形面积，再判断是否为0.

def func2(p1, p2, p3):
    return (1/2) * (p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p2[0]*p1[1] - p3[0]*p2[1])

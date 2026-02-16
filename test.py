from functools import reduce

# 定义管道函数：接收函数列表和初始值，自动串联执行
def pipe(functions, initial_value):
    return reduce(lambda x, f: f(x), functions, initial_value)

# 定义步骤函数
def add1(x): return x + 1
def mul2(x): return x * 2
def sub3(x): return x - 3

# 管道式执行：1 +1 → 2 ×2 →4 -3 →1
result = pipe([add1, mul2, sub3], 1)
print(result)  # 输出：1
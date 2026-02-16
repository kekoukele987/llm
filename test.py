# 普通列表（低效）
import torch


a = [[1,2], [3,4]]
b = [[5,6], [7,8]]
# 手写矩阵乘法（麻烦）
result = [[a[0][0]*b[0][0]+a[0][1]*b[1][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]],
          [a[1][0]*b[0][0]+a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]]]

# 张量（高效）
a_tensor = torch.tensor(a)
b_tensor = torch.tensor(b)
result_tensor = torch.matmul(a_tensor, b_tensor)  # 一行搞定
print(result_tensor)
# 输出：
# tensor([[19, 22],
#         [43, 50]])
import matplotlib.pyplot as plt
import numpy as np

fr = open("train_log/Train2_1/log_2023-09-26-16-30-57.txt")

step = 0
data = [[] for _ in range(4)]
for l in fr.readlines()[:120]:
    s = l.split("\t")
    step += 1
    data[0].append(float(s[8]))
    data[1].append(float(s[9]))
    data[2].append(float(s[6]))

newdata = []
for d in data:
    d = np.array(d)
    newdata.append((d[2:] + d[1:-1] + d[:-2]) / 3)
data = newdata

step = np.arange(0, 120 - 2, 3 )
plt.xlabel('step')
plt.ylabel('Error')
plt.plot(step, data[0][::3], 'ro-', label='Trajectory')
plt.plot(step, data[1][::3], 'g^-', label='Action')
#plt.plot(step, data[2][::3], 'b^-', label='m₃')
#plt.plot(step, data[0][::3], 'r-', label='Trajectory')
plt.legend()
plt.savefig('likelihood.png', dpi=200)
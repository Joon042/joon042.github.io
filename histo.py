import matplotlib.pyplot as plt

score = [ '0/8', '1/8', '2/8', '3/8', '4/8', '5/8', '6/8', '7/8', '8/8' ]

count = [37, 52, 29, 36, 184, 65, 9489, 108, 0]

bar = plt.bar(score, count)
plt.xlabel("score")
plt.ylabel("count")
plt.title("bar graph dmax=10,000")
plt.bar_label(bar)
plt.savefig("bar_dmax=10,000.png")

import matplotlib.pyplot as plt
from PIL import Image



W, H = 1904, 988

img = Image.open("./data/yke_data/split_2sec_1500mHz/images/video_1_frame_0001.png")  # path to your image
X = [1008.35, 453.3, 978.05, 1013.0, 1579.05, 952.5, 954.05]
Y = [244.0, 1029.2, 858.75, 269.4, 200.0, 909.25, 848.25]
T = [491.5, 185.0, 191.0, 261.5, 113.0, 103.0, 123.0]

# img = Image.open("./data/yke_data/split_2sec_1500mHz/images/video_1_frame_1201.png")  # path to your image
# # Scanpath data
# X = [1561.5, 938.7, 409.6, 1484.7, 1568.8, 1461.75]
# Y = [756.8, 812.5, 176.5, 822.05, 872.65, 885.0]
# T = [914.25, 260.5, 196.5, 167.0, 251.5, 7.25]


sizes = [t * 2 for t in T]

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(img)
ax.plot(X, Y, linewidth=1.5)
ax.scatter(X, Y, s=sizes, alpha=0.6)

for i, (x, y) in enumerate(zip(X, Y), start=1):
    ax.text(x, y, str(i), fontsize=10, ha="center", va="center")

ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_title("Scanpath on image")
plt.show()
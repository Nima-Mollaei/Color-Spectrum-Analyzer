import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

# تعداد رنگ‌های غالبی که می‌خواییم نشون بدیم
NUM_CLUSTERS = 10

# مرحله 1: بارگذاری عکس
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تغییر شکل عکس به آرایه‌ی دو بعدی
reshaped_image = image.reshape((-1, 3))

# مرحله 2: خوشه‌بندی رنگ‌ها با KMeans
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(reshaped_image)

# گرفتن رنگ‌های غالب
colors = kmeans.cluster_centers_.astype(int)
counts = Counter(kmeans.labels_)

# مرحله 3: رسم طیف رنگی
def plot_color_bar(colors, counts):
    total = sum(counts.values())
    ratios = [count / total for count in counts.values()]
    
    bar = np.zeros((50, 300, 3), dtype='uint8')
    start = 0
    for ratio, color in zip(ratios, colors):
        end = start + int(ratio * 300)
        bar[:, start:end, :] = color
        start = end
    return bar

color_bar = plot_color_bar(colors, counts)

# نمایش عکس اصلی و طیف رنگی
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(color_bar)
plt.title('Color Spectrum')
plt.axis('off')

plt.tight_layout()
plt.show()


import cv2
import numpy as np

image = cv2.imread('E:\\code\\img\\res-mask.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)   # 原图

res = np.zeros((512, 512))
position = np.argwhere(edges == 255)
for point in position:
    i = point[0]
    j = point[1]
    for v in range(4):
        if j-v >= 0 and j+v < 512:
            edges[i][j-v] = 255
            edges[i][j+v] = 255

seed_point = (100, 100)


def region_growing(image, seed_point, threshold):
    visited = set()
    region = []
    region.append(seed_point)
    
    while len(region) > 0:
        current_point = region.pop()
        visited.add(current_point)
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = current_point[0] + dx
                y = current_point[1] + dy
                
                if (x, y) not in visited and 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    if abs(int(image[x, y]) - int(image[current_point])) < threshold:
                        region.append((x, y))
                        visited.add((x, y))
                        image[x, y] = 0
    
    return image

threshold = 10
filled_image = region_growing(edges, seed_point, threshold)


cv2.imwrite('E:\\code\\img\\res-mask-4.png', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

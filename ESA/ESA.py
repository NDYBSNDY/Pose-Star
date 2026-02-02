import cv2
import numpy as np

image = cv2.imread('E:\\code\\ESA\\img\\img-line.png')
image2 = cv2.imread('E:\\code\\ESA\\img\\mask-line.png')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 100, 200)   # 原图
edges2 = cv2.Canny(gray2, 100, 200) # 注意力图


res = np.zeros((512, 512))
position = np.argwhere(edges2 == 255)
for point in position:
    i = point[0]
    j = point[1]
    tag = 0
    if edges[i][j] == 255:
        tag = 1
        res[i][j] = 255
        continue
    else:

        for v in range(70):
            if j-v >= 0 and j+v < 512:
                if edges[i][j-v] == 255:
                    tag = 1
                    res[i][j-v] = 255
                    # continue
                    break
                elif edges[i][j+v] == 255:
                    tag = 1
                    res[i][j+v] = 255
                    # continue
                    break
    if tag == 0:
        res[i][j] = edges2[i][j]

#################################


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

cv2.imwrite('E:\\code\\edges-line-6.png', res)
cv2.waitKey(0)
cv2.destroyAllWindows()



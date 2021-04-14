import cv2

a = cv2.imread('./chessboard_hp.jpg')
print(a.shape)
b = cv2.resize(a, (int(a.shape[1]/4), (int(a.shape[0]/4))))
print(b.shape)

cv2.imshow('a',a)
cv2.imshow('b', b)
cv2.waitKey(0)
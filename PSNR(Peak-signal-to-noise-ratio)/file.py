import cv2
original1 = '' #file path
compressed1 = '' #file path


img = cv2.imread(original1, cv2.IMREAD_UNCHANGED)
width = 500
height = 500
dim = (width, height)
original = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('original',original)
cv2.waitKey()


compressed = cv2.imread(compressed1,1)
width = 500
height = 500
dim = (width, height)
compressed = cv2.resize(compressed , dim, interpolation = cv2.INTER_AREA)
cv2.imshow(compressed) 
cv2.waitKey()

value = PSNR(original, compressed)
print(f"PSNR value is {value} dB") 


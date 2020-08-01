import cv2
import numpy as np

def TP_matching(file1,file2):
    img1 = cv2.imread('./Fidelity/' + file1,0);
    img2 = cv2.imread('./Fidelity/' + file2,0);
    h,w = img1.shape[0:2]
    error_sum = 0
    for row in range(h):
        for col in range(w):
            error_sum += pow(int(img1[row,col])-int(img2[row,col]),2)
    error_sum = pow(error_sum/(h*w),0.5)
    return error_sum

if __name__ == '__main__':
    rms = TP_matching('01.png','02.png')
    print('01.png与02.png的均方根误差为'+str(rms))
    rms = TP_matching('01.png', '03.png')
    print('01.png与03.png的均方根误差为' + str(rms))

# def margin_off(img):
#     toprow= 0
#     while True:
#         rowdata = img[toprow,:]
#         if(np.mean(rowdata)>240):
#             toprow+=1
#         else:
#             break;
#     print(min(rowdata))
#     botrow = img.shape[0]-1
#     while True:
#         rowdata = img[botrow, :]
#         if (np.mean(rowdata)>240):
#             botrow -= 1
#         else:
#             break;
#
#     leftcol = 0
#     while True:
#         coldata = img[:,leftcol]
#         if (np.mean(coldata)>240):
#             leftcol += 1
#         else:
#             break;
#
#     rightcol = img.shape[1] - 1
#     while True:
#         coldata = img[:,rightcol]
#         if (np.mean(coldata)>240):
#             rightcol -= 1
#         else:
#             break;
#     return img[toprow:botrow,leftcol:rightcol]
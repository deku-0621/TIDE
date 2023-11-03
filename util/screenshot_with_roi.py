import cv2

def take_screenshot_with_roi(in_path,file_path,reshape=False):
    # 初始化摄像头
    if  not reshape:
     img = cv2.imread(in_path)
    else:
     img = cv2.resize(cv2.imread(in_path),(480,640))

    # 使用cv2.selectROI函数交互式地选择感兴趣区域
    roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=False)
    # 从选择的ROI中截取图像
    roi_cropped = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    # 将截取的图像保存到指定路径
    cv2.imwrite(file_path, roi_cropped)

    cv2.destroyAllWindows()  # 关闭显示窗口
    return roi

    print(f"截图已保存到：{file_path}")

if __name__ == "__main__":
    in_path = "../tests/input/93406.jpg"
    file_path = "../tests/input/93406_s3.jpg"
    take_screenshot_with_roi(in_path,file_path,False)

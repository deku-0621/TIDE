import cv2
import numpy as np


def extract_foreground(image_path, roi_coordinates):
    # 读取图像
    image = cv2.imread(image_path)

    # 确保roi_coordinates是整数
    x, y, w, h = map(int, roi_coordinates)

    # 提取ROI
    roi = image[y:y + h, x:x + w]

    # 进行前景提取
    # 在此我们使用GrabCut算法，可以根据需要尝试其他算法
    mask = np.zeros(roi.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, roi.shape[1] - 10, roi.shape[0] - 10)  # 设置一个边界矩形
    cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = roi * mask2[:, :, np.newaxis]

    return result


if __name__ == "__main__":
    from screenshot_with_roi import take_screenshot_with_roi

    in_path = "tests/input/gc2.jpg"
    file_path = "tests/input/gc3.jpg"
    roi_coordinates = take_screenshot_with_roi(in_path,file_path,True)
    result_image = extract_foreground(file_path, roi_coordinates)

    # 显示结果
    cv2.imwrite(in_path, result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

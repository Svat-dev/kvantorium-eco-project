# import cv2

# cap = cv2.VideoCapture("./assets/images/image.jpg")


def intersection_area(rect_a, rect_b):
    x1a, y1a, x2a, y2a = rect_a
    x1b, y1b, x2b, y2b = rect_b

    x_intersect = max(x1a, x1b)
    y_intersect = max(y1a, y1b)
    w_intersect = min(x2a, x2b) - x_intersect
    h_intersect = min(y2a, y2b) - y_intersect

    # success, img = cap.read()

    # cv2.rectangle(img, (x1a, y1a), (x2a, y2a), (0, 255, 0), 2)
    # cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if w_intersect > 0 and h_intersect > 0:
        return w_intersect * h_intersect
    else:
        return 0


def rectangle_overlap_percentage(rect_a, rect_b):
    area_a = abs((rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1]))
    area_b = abs((rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1]))

    intersect_area = intersection_area(rect_a, rect_b)
    total_area = area_a + area_b - intersect_area

    overlap_ratio = intersect_area / total_area if total_area != 0 else 0
    return overlap_ratio


def get_the_biggest():
    maximum = 0
    with open("output.txt", "r") as file:
        content = file.read().split("\n")
        for line in content:
            if line:
                value = float(line.replace("%", ""))
                maximum = max(maximum, value)
    with open("output.txt", "a") as file:
        file.write(f"Max value: {maximum}%\n")


# # Пример использования
# rect_a = (
#     100,
#     100,
#     300,
#     300,
# )  # Прямоугольник A: левый верх (100,100), правый низ (300,300)
# rect_b = (
#     200,
#     200,
#     400,
#     400,
# )  # Прямоугольник B: левый верх (200,200), правый низ (400,400)

# print(f"Перекрытие (%): {rectangle_overlap_percentage(rect_a, rect_b) * 100:.2f}%")

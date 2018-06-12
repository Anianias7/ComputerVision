from matplotlib import pyplot as plt


def show_images_with_points(image, points_coords, first_sift):
    plt.imshow(image)
    plt.plot(([points[0][0] for points in points_coords],
              [points[1][0] for points in points_coords]),
             ([points[0][1] for points in points_coords],
              [points[1][1] + first_sift[2].shape[0] for points in points_coords]))
    plt.show()

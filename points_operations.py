import cv2 as cv
import numpy as np


def get_key_points_and_features(img_name):
    img = cv.imread(img_name)
    color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(color, None)
    return kp, np.asarray(des), color


def compute_distances_for_image_key_point(key_point_features, img_features):
    return np.sum(np.abs((img_features - key_point_features.reshape((1, 128)))), axis=1)


def find_key_point_nearest_neighbour(list_of_distances):
    return np.argmin(list_of_distances)


def get_neighbours(fst_image, snd_image):
    point_neighbour_list = []
    for index, value in enumerate(fst_image):
        distances = compute_distances_for_image_key_point(value, snd_image)
        neighbour_index = find_key_point_nearest_neighbour(distances)
        point_neighbour_list.append((index, neighbour_index))

    for index, value in enumerate(snd_image):
        distance = compute_distances_for_image_key_point(value, fst_image)
        neighbour_index = find_key_point_nearest_neighbour(distance)
        point_neighbour_list.append((neighbour_index, index))
    return point_neighbour_list


def get_pairs_of_key_points(list_of_neighbours):
    return [x for (i, x) in enumerate(list_of_neighbours) if x in list_of_neighbours[i + 1:]]


def get_coords_of_points_pairs(fst_image_points, snd_image_points, list_of_pairs):
    return [[(fst_image_points[fst_index].pt[0], fst_image_points[fst_index].pt[1]),
             (snd_image_points[snd_index].pt[0], snd_image_points[snd_index].pt[1])]
            for (fst_index, snd_index) in list_of_pairs]


def point_n_neighbors_indexes(point_features, image_features, num_of_neighbours):
    distances = compute_distances_for_image_key_point(point_features, image_features)
    return np.argsort(distances)[1:num_of_neighbours]


def find_neighbors_for_key_pair(pair, img1, img2, num_of_neighbors):
    fst_index = pair[0]
    snd_index = pair[1]
    fst_indexes = point_n_neighbors_indexes(img1[fst_index].reshape((1, 128)), img1, num_of_neighbors)
    snd_indexes = point_n_neighbors_indexes(img2[snd_index].reshape((1, 128)), img2, num_of_neighbors)
    return (fst_index, fst_indexes), (snd_index, snd_indexes)


def get_key_points_pairs_and_its_neighbors(list_of_key_points_pairs, img1, img2, num_of_neighbors):
    return [find_neighbors_for_key_pair(pair, img1, img2, num_of_neighbors) for pair in
            list_of_key_points_pairs]


# return (img1_index, img2_index), num_of_cohesion
def count_cohesion_of_the_neighborhood_of_key_points(key_points_pair_and_its_neighbors, list_of_key_points_pairs):
    num_of_cohesion = 0
    for key_points_pair in list_of_key_points_pairs:
        if (key_points_pair[0] in key_points_pair_and_its_neighbors[0][1]) and (
                key_points_pair[1] in key_points_pair_and_its_neighbors[1][1]):
            num_of_cohesion += 1
    return (key_points_pair_and_its_neighbors[0][0], key_points_pair_and_its_neighbors[1][0]), num_of_cohesion


def get_points_and_its_cohesion(list_of_key_points_pairs_and_its_neighbors, list_of_key_points_pairs):
    return [count_cohesion_of_the_neighborhood_of_key_points(pair, list_of_key_points_pairs) for pair in
            list_of_key_points_pairs_and_its_neighbors]


def get_points_with_more_than_min_cohesion(list_of_points_and_its_cohesion, min_cohesion):
    return [coords for (coords, coh) in list_of_points_and_its_cohesion if coh > min_cohesion]


def remove_duplicates(list_of_points_pairs_indexes):
    result = []
    for pair in list_of_points_pairs_indexes:
        if pair not in result:
            result.append(pair)
    return np.asarray(result)



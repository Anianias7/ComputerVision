import random

import cv2
import numpy as np


def get_n_random_values_from_list(given_list, n):
    random.shuffle(given_list)
    return given_list[:n]


def adjust_list_of_key_pair_for_affine_transformation_input(list_of_key_points_pairs_coords):
    fst_img_points_coords = [[fst_image_coords[0], fst_image_coords[1]] for
                             fst_image_coords, _ in list_of_key_points_pairs_coords]
    snd_img_points_coords = [[snd_image_coords[0], snd_image_coords[1]] for
                             _, snd_image_coords in list_of_key_points_pairs_coords]
    return np.float32(fst_img_points_coords), np.float32(snd_img_points_coords)


def calculate_affine_transformation(key_pair_for_transformation):
    result = cv2.getAffineTransform(key_pair_for_transformation[0],
                                    key_pair_for_transformation[1])
    return result


def calculate_perspective_transformation(key_pair_for_transformation):
    result = cv2.getPerspectiveTransform(key_pair_for_transformation[0],
                                         key_pair_for_transformation[1])
    return result


def calculate_transform_point_coords(transformation, point_coords):
    return transformation @ point_coords


# def model_error(model, key_point_pair_coords):
#     fst_img_point_coords = np.append(np.asarray(key_point_pair_coords[0]), 1).reshape([3, 1])
#     snd_img_point_coords = np.append(np.asarray(key_point_pair_coords[1]), 1).reshape([3, 1])
#     transform_point = np.append(calculate_transform_point_coords(model, fst_img_point_coords), 1).reshape([3, 1])
#     return np.linalg.norm(transform_point - snd_img_point_coords, ord=2)


def model_error(model, key_point_pair_coords):
    fst_img_point_coords = np.append(np.asarray(key_point_pair_coords[0]), 1).reshape([3, 1])
    snd_img_point_coords = np.append(np.asarray(key_point_pair_coords[1]), 1).reshape([3, 1])
    transform_point = calculate_transform_point_coords(model, fst_img_point_coords)
    return np.linalg.norm(transform_point - snd_img_point_coords, ord=2)


# def affine_transformation(key_pair_for_transformation):
#     return np.linalg.inv(create_point_matrix(key_pair_for_transformation[0])) @ create_points_on_snd_img_vector(
#         key_pair_for_transformation[1])
#
#
# def create_points_on_snd_img_vector(list_of_points):
#     x_coords = np.asarray([x for x, y in list_of_points])
#     y_coords = np.asarray([y for x, y in list_of_points])
#     return np.concatenate((x_coords, y_coords)).reshape([6, 1])


def create_point_matrix(list_of_points):
    print(list_of_points)
    result = None
    for point in list_of_points:
        x = point[0]
        y = point[1]
        row = np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ])
        result = np.concatenate((result, row), axis=0) if result is not None else row
    return result


def ransac(list_of_key_points_pairs_coords, num_of_iteration, num_of_samples, max_error):
    best_model = None
    best_score = 0
    best_inlainers = []
    for i in range(0, num_of_iteration):
        inliners = []
        score = 0
        sample = get_n_random_values_from_list(list_of_key_points_pairs_coords, num_of_samples)
        coords_of_points = adjust_list_of_key_pair_for_affine_transformation_input(sample)
        # model = calculate_affine_transformation(coords_of_points)
        model = calculate_perspective_transformation(coords_of_points)
        for pair in list_of_key_points_pairs_coords:
            error = model_error(model, pair)
            print(error)
            if error < max_error:
                score += 1
                inliners.append(pair)
        if score > best_score:
            best_score = score
            best_model = model
            best_inlainers = inliners
    return best_model, best_inlainers

from timeit import default_timer as timer
import datetime

from display import show_images_with_points

from points_operations import *
from ransac import ransac

# fst_image = "s21.jpg"
# snd_image = "s22.jpg"

fst_image = "dd1.jpg"
snd_image = "dd2.jpg"


first_sift = get_key_points_and_features(fst_image)
second_sift = get_key_points_and_features(snd_image)

image = np.concatenate((first_sift[2], second_sift[2]))
neighbours = get_neighbours(first_sift[1], second_sift[1])
points_pairs = get_pairs_of_key_points(neighbours)
# points_coords = get_coords_of_points_pairs(first_sift[0], second_sift[0], points_pairs)

#COHESION
key_points_pairs_and_its_neighbors = get_key_points_pairs_and_its_neighbors(points_pairs, first_sift[1], second_sift[1],
                                                                            50)
points_with_cohesion = get_points_and_its_cohesion(key_points_pairs_and_its_neighbors, points_pairs)
points_with_accepted_cohesion = get_points_with_more_than_min_cohesion(points_with_cohesion, 10)
points_coords = get_coords_of_points_pairs(first_sift[0], second_sift[0], points_with_accepted_cohesion)

# #RANSAC:
start_time = datetime.datetime.now()
print(start_time)
# points_coords = get_coords_of_points_pairs(first_sift[0], second_sift[0], points_pairs)
get_points_after_transformation = ransac(points_coords, 100, 4, 50)[1]
end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print(end_time)
print(elapsed_time.seconds)
show_images_with_points(image, get_points_after_transformation, first_sift)

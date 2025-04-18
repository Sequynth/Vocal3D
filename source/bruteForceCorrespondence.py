import helper
import numpy as np
from RHC import globalAlignment


def calc_error(camera, points_cameraspace, laser, points_gridspace):
    error = 0.0

    cameraRays = camera.getRayMat(np.flip(points_cameraspace, axis=1)) # The error is here!
    laserRays = laser.rays()[laser.getNfromXY(points_gridspace[:, 0], points_gridspace[:, 1])]
    origin = np.expand_dims(laser.origin(), 0)

    aPoints, bPoints, distances = helper.MatLineLineIntersection(cameraRays*0.0, cameraRays*100.0, origin + 0.0*laserRays, origin + 100.0*laserRays)
    reprojections = helper.project3DPointToImagePlaneMat(aPoints + ((bPoints - aPoints) / 2.0), camera.intrinsic())
    error = np.sum(np.linalg.norm(reprojections - np.flip(points_cameraspace, axis=1), axis=1))

    return error*error

def compute(laserGridStuff, pixelLocations, laserpoint_image, camera, laser, set_size, iterations):
    grid2DPixLocations = globalAlignment(laserGridStuff, pixelLocations, laserpoint_image, laser)

    gridIDs = np.array([x for x, _ in grid2DPixLocations])
    points2D = np.array([x for _, x in grid2DPixLocations])
    
    min_x = gridIDs[:, 0].min()
    max_x = gridIDs[:, 0].max()
    min_y = gridIDs[:, 1].min()
    max_y = gridIDs[:, 1].max()

    smallest_errors = []
    update_vectors = []


    for i in range(iterations):
        random_indices = np.random.choice(len(gridIDs), set_size, replace=False)
        grid_samples = gridIDs[random_indices]
        point_samples = points2D[random_indices]

        min_error = 10000
        update_vector = np.array([0, 0])
        for x in range(-min_x, laser.gridWidth() - max_x):
            for y in range(-min_y, laser.gridHeight() - max_y):
                new_grid_samples = grid_samples + np.array([x, y], dtype=int)
                error = calc_error(camera, point_samples, laser, new_grid_samples)
                if error < min_error:
                    min_error = error
                    update_vector = np.array([x, y], dtype=int)
        smallest_errors.append(min_error)
        update_vectors.append(update_vector)

    best_update_vector = update_vectors[smallest_errors.index(min(smallest_errors))]
    
    grid2DPixLocations = []
    for i in range(gridIDs.shape[0]):
        grid2DPixLocations.append(
            [gridIDs[i] + best_update_vector, points2D[i]]
        )
    
    return grid2DPixLocations
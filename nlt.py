import cv2
import numpy as np


def calculate_extrinsic_parameters(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rvecs = np.zeros((len(object_points), 3))
    tvecs = np.zeros((len(object_points), 3))

    if len(image_points.shape) == 3:
        image_points = image_points.reshape(image_points.shape[0], image_points.shape[1], 1, image_points.shape[2]).astype(np.float32)
    if len(object_points.shape) == 3:
        object_points = object_points.reshape(object_points.shape[0], object_points.shape[1], 1, object_points.shape[2]).astype(np.float32)


    for i in range(len(object_points)):
        _, rvec, tvec = cv2.solvePnP(
            object_points[i],
            image_points[i],
            camera_matrix,
            distortion_coefficients,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        rvecs[i] = rvec.reshape(3)
        tvecs[i] = tvec.reshape(3)

    return rvecs, tvecs

def calculate_extrinsic_parameters_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rvecs = np.zeros((len(object_points), 3))
    tvecs = np.zeros((len(object_points), 3))

    if len(image_points.shape) == 3:
        image_points = image_points.reshape(image_points.shape[0], image_points.shape[1], 1, image_points.shape[2]).astype(np.float32)
    if len(object_points.shape) == 3:
        object_points = object_points.reshape(object_points.shape[0], object_points.shape[1], 1, object_points.shape[2]).astype(np.float32)

    for i in range(len(object_points)):
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points[i],
            image_points[i],
            camera_matrix,
            distortion_coefficients,
            useExtrinsicGuess=False,
            iterationsCount=10000,
            reprojectionError=1.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        rvecs[i] = rvec.reshape(3)
        tvecs[i] = tvec.reshape(3)

    return rvecs, tvecs


def calculate_world_coordinates(
    c1_rvec: np.ndarray,
    c1_tvec: np.ndarray,
    c1_camera_matrix: np.ndarray,
    c1_image_point: np.ndarray,
    c2_rvec: np.ndarray,
    c2_tvec: np.ndarray,
    c2_camera_matrix: np.ndarray,
    c2_image_point: np.ndarray,
) -> np.ndarray:
    # Convert input data to numpy arrays of type float32 for consistency and compatibility with OpenCV functions
    c1_rvec = np.array(c1_rvec, dtype=np.float32)
    c1_tvec = np.array(c1_tvec, dtype=np.float32)
    c1_camera_matrix = np.array(c1_camera_matrix, dtype=np.float32)
    c1_image_point = np.array(c1_image_point, dtype=np.float32)
    c2_rvec = np.array(c2_rvec, dtype=np.float32)
    c2_tvec = np.array(c2_tvec, dtype=np.float32)
    c2_camera_matrix = np.array(c2_camera_matrix, dtype=np.float32)
    c2_image_point = np.array(c2_image_point, dtype=np.float32)

    if len(c1_image_point.shape) == 2:
        c1_image_point = c1_image_point.reshape(-1, 1, 2)
    if len(c2_image_point.shape) == 2:
        c2_image_point = c2_image_point.reshape(-1, 1, 2)

    # Convert rotation vectors to rotation matrices using Rodrigues' transformation
    c1_rot_mat, _ = cv2.Rodrigues(c1_rvec)
    c2_rot_mat, _ = cv2.Rodrigues(c2_rvec)

    # Construct camera projection matrices from the rotation matrices and translation vectors
    P1 = c1_camera_matrix @ np.hstack((c1_rot_mat, c1_tvec.reshape(3, 1)))
    P2 = c2_camera_matrix @ np.hstack((c2_rot_mat, c2_tvec.reshape(3, 1)))

    # Triangulate the 3D world coordinates from the 2D image points in each camera view
    point4D = cv2.triangulatePoints(P1, P2, c1_image_point, c2_image_point)

    # Convert the homogeneous coordinates to 3D coordinates by normalizing with the fourth row
    world_coordinates = point4D[:3] / point4D[3]

    # Return the transposed matrix to align with the conventional shape for coordinates
    return world_coordinates.T


def MPJPE(predicted_points: np.ndarray, true_points: np.ndarray) -> float:
    # Ensure that 'predicted_points' and 'true_points' are numpy arrays with the shape (n_points, 3)
    assert predicted_points.shape == true_points.shape, "Shapes must match"

    # Calculate the Euclidean distance for each corresponding pair of points
    distances = np.linalg.norm(predicted_points - true_points, axis=1)

    # Compute the average of these distances
    mpjpe = np.mean(distances)
    return mpjpe

def sigmoid(x: np.ndarray, k: int = 50) -> np.ndarray:
    weights = 1 / (1 + np.exp(-k * (x - np.mean(x))))
    normalized_weights = weights / np.sum(weights)
    return normalized_weights
import cv2
import math
import numpy as np
import pyrealsense2 as rs


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    # points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()
    points = np.transpose(position[0:3, :])

    return points


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
print('Device detected.', device)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
# width of the image in pixels
width = depth_intrinsics.width
# height of the image in pixels
height = depth_intrinsics.height
# horizontal coordinate of the principal point of the image, 
# as a pixel offset from the left edge
cx = depth_intrinsics.ppx
# vertical coordinate of the principal point of the image, 
# as a pixel offset from the top edge
cy = depth_intrinsics.ppy
# focal length of the image plane, as a multiple of pixel width
fx = depth_intrinsics.fx
# focal length of the image plane, as a multiple of pixel height
fy = depth_intrinsics.fy
# distortion model of the image
distortion_model = depth_intrinsics.model
# distortion coefficients
distortion_coeffs = depth_intrinsics.coeffs

config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
Kinv = np.linalg.inv(K)
# print(Kinv @ np.array([[400], [800], [1]]))

data_idx = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        camera_coordinates = np.zeros((height * width, 3))
        camera_coordinates = []
        for i in range(height):
            for j in range(width):
                cur_depth = 0.001 * depth_image[i][j]
                if cur_depth != 0 and cur_depth <= 2:
                    # cur_pixel_coordinate = np.array([[i], [j], [1]])
                    # cur_camera_coordinate_norm = Kinv @ cur_pixel_coordinate
                    # cur_camera_coordinate = 0.001 * depth_image[i][j] * cur_camera_coordinate_norm
                    Z = cur_depth
                    X = Z * (i - cx) / fx
                    Y = Z * (j - cy) / fy
                    # camera_coordinates.append(cur_camera_coordinate.reshape(3,))
                    camera_coordinates.append(np.array([X, Y, Z]))
        camera_coordinates = np.array(camera_coordinates)

        # points = depth_image_to_point_cloud(color_image, depth_image, 1000, K, np.eye(4))
        # print(points)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break
        
        if key == ord("e"):
            np.save('./data/real_data/data' + str(data_idx) + '.npy', camera_coordinates)
            # np.save('./data/real_data/data' + str(data_idx) + '.npy', points)
            data_idx += 1

finally:

    # Stop streaming
    pipeline.stop()
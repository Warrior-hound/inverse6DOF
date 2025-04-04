import numpy as np
import math

l1_length = 35
l2_length = 35
l3_length = 35
l4_length = 35
l5_length = 35
l6_length = 30

def tan_inverse(value):
    return math.degrees(math.atan(value))

def cos_inverse(value):
    return math.degrees(math.acos(value))

def cos(angle):
    return np.cos(np.radians(angle))

def sin(angle):
    return np.sin(np.radians(angle))

def sec(angle):
    return 1 / np.cos(np.radians(angle))

def tan(angle):
    return 1 / np.tan(np.radians(angle))


def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Rotation matrix for roll (about x-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation matrix for pitch (about y-axis)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation matrix for yaw (about z-axis)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Final rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

def get_full_transformation_matrix(roll, pitch, yaw, x, y, z):
    # Step 1: Get rotation matrix from Euler angles
    rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # Step 2: Create translation vector
    translation_vector = np.array([x, y, z])

    # Step 3: Combine rotation matrix and translation vector into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Create a 4x4 identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Top-left 3x3 is the rotation matrix
    transformation_matrix[:3, 3] = translation_vector  # Top-right 3x1 is the translation vector
    
    return transformation_matrix

# Example usage:
roll = 0  # degrees
pitch = 90  # degrees
yaw = 3 # degrees
x = 135  # translation along X-axis
y = 0   # translation along Y-axis
z = 70   # translation along Z-axis

full_transformation_matrix = get_full_transformation_matrix(roll, pitch, yaw, x, y, z)
l56 = l5_length + l6_length
print("Full Transformation Matrix:")
print(full_transformation_matrix)

end_effector_position = [x,y,z]
wrist_position = [
                x - (l56 * full_transformation_matrix[0][2]),
                y - (l56 * full_transformation_matrix[1][2]),
                z - (l56 * full_transformation_matrix[2][2])
               ]

print(end_effector_position)
print(wrist_position)

theta_1 = tan_inverse(wrist_position[1] / wrist_position[0])
print(theta_1)
r = (math.sqrt((wrist_position[0] ** 2) + (wrist_position[1] ** 2))) - l2_length

s = wrist_position[2] - l1_length


hypo_theta = tan_inverse(s / r)
remaining_theta = 90 - hypo_theta
hypotenuse = r / cos(hypo_theta)


theta_2_term = ((l3_length ** 2) + (hypotenuse ** 2) - (l4_length ** 2)) / (2 * l3_length * hypotenuse)

theta_2 = cos_inverse(theta_2_term) - remaining_theta
print((-1 * theta_2))
theta_3_term = ((l3_length ** 2) + (l4_length ** 2) - (hypotenuse ** 2)) / (2 * l3_length * l4_length)

theta_3 = 90-cos_inverse(theta_3_term)



print(theta_3)

theta4_term_1 = (-cos(theta_1)*sin(theta_2+theta_3)*full_transformation_matrix[0][0]) - (sin(theta_1)*sin(theta_2+theta_3)*full_transformation_matrix[1][0]) + (cos(theta_2+theta_3)*full_transformation_matrix[1][0])
theta4_term_2 = (cos(theta_1)*cos(theta_2+theta_3)*full_transformation_matrix[0][0]) + (sin(theta_1)*cos(theta_2+theta_3)*full_transformation_matrix[1][0]) + (sin(theta_2+theta_3)*full_transformation_matrix[1][0])

theta_4 = tan_inverse(theta4_term_1 / theta4_term_2)
print(theta_4)

theta_5_term_1 = math.sqrt(1 - ((sin(theta_1)*full_transformation_matrix[0][2])-(cos(theta_1)*full_transformation_matrix[1][2]))**2)
theta_5_term_2 = (sin(theta_1)*full_transformation_matrix[0][2]) - (cos(theta_1)*full_transformation_matrix[1][2])

if abs(theta_5_term_2) < 1e-6:
    print("Warning: theta_5_term_2 is close to zero. Using a default value.")
    theta_5 = 0  # Default or special case for theta_5
else:
    theta_5 = tan_inverse(theta_5_term_1 / theta_5_term_2) - 90

print(theta_5)

theta_6_term_1 = (sin(theta_1)*full_transformation_matrix[0][1]) - (cos(theta_1)*full_transformation_matrix[1][1])
theta_6_term_2 = (-sin(theta_1)*full_transformation_matrix[0][0]) + (cos(theta_1)*full_transformation_matrix[1][0])


if abs(theta_6_term_2) < 1e-6:
    print("Warning: theta_5_term_2 is close to zero. Using a default value.")
    theta_6 = 0  # Default or special case for theta_5
else:
    theta_6 = tan_inverse(theta_6_term_1 / theta_6_term_2) - 90


print(theta_6)

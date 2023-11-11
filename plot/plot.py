import matplotlib.pyplot as plt


def plot_loss(hist_loss):
    fig, ax = plt.subplots()
    ax.plot(hist_loss)


def plot_skeleton_3d(joint_pos, joint_names):
    bone_list = [['SPINE_CHEST', 'SPINE_NAVEL'],
     ['SPINE_NAVEL', 'PELVIS'],
     ['SPINE_CHEST', 'NECK'],
     ['NECK', 'HEAD'],
     ['HEAD', 'NOSE'],
     ['SPINE_CHEST', 'CLAVICLE_LEFT'],
     ['CLAVICLE_LEFT', 'SHOULDER_LEFT'],
     ['SHOULDER_LEFT', 'ELBOW_LEFT'],
     ['ELBOW_LEFT', 'WRIST_LEFT'],
     ['WRIST_LEFT', 'HAND_LEFT'],
     ['HAND_LEFT', 'HANDTIP_LEFT'],
     ['WRIST_LEFT', 'THUMB_LEFT'],
     ['PELVIS', 'HIP_LEFT'],
     ['HIP_LEFT', 'KNEE_LEFT'],
     ['KNEE_LEFT', 'ANKLE_LEFT'],
     ['ANKLE_LEFT', 'FOOT_LEFT'],
     ['NOSE', 'EYE_LEFT'],
     ['EYE_LEFT', 'EAR_LEFT'],
     ['SPINE_CHEST', 'CLAVICLE_RIGHT'],
     ['CLAVICLE_RIGHT', 'SHOULDER_RIGHT'],
     ['SHOULDER_RIGHT', 'ELBOW_RIGHT'],
     ['ELBOW_RIGHT', 'WRIST_RIGHT'],
     ['WRIST_RIGHT', 'HAND_RIGHT'],
     ['HAND_RIGHT', 'HANDTIP_RIGHT'],
     ['WRIST_RIGHT', 'THUMB_RIGHT'],
     ['PELVIS', 'HIP_RIGHT'],
     ['HIP_RIGHT', 'KNEE_RIGHT'],
     ['KNEE_RIGHT', 'ANKLE_RIGHT'],
     ['ANKLE_RIGHT', 'FOOT_RIGHT'],
     ['NOSE', 'EYE_RIGHT'],
     ['EYE_RIGHT', 'EAR_RIGHT']]
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Determine which coordinate goes to which axis in the figure
    x, z, y = range(3)

    # Invert the z axis
    ax.invert_zaxis()

    joint_names = list(joint_names)

    for bone in bone_list:
        idx_joint_0 = joint_names.index(bone[0])
        idx_joint_1 = joint_names.index(bone[1])
        ax.plot([joint_pos[idx_joint_0][x], joint_pos[idx_joint_1][x]],
                [joint_pos[idx_joint_0][y], joint_pos[idx_joint_1][y]],
                [joint_pos[idx_joint_0][z], joint_pos[idx_joint_1][z]], color='b')

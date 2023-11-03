import torch
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

X_AXIS = torch.from_numpy(np.array([1, 0, 0])).float()

def normalize(v):
    norm = torch.norm(v, p=None, dim=2, keepdim=True)
    return v / norm


def get_quat_rot_between(u, v, eps=1e-3):
    """
    NOTE: This version assumes that u and v are normalised!
    Otherwise the "k = " line needs to use dot instead of norm.
    See: https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    """
    k_cos_theta = torch.bmm(u.reshape((u.shape[0] * u.shape[1], 1, u.shape[2])),
                            v.reshape((v.shape[0] * v.shape[1], v.shape[2], 1))).reshape((u.shape[0], u.shape[1], 1))
    k = torch.sqrt(torch.norm(u, p=None, dim=2, keepdim=True)
                   * torch.norm(v, p=None, dim=2, keepdim=True))
    quat = torch.zeros((u.shape[0], v.shape[1], 4))

    # TODO: this version does no check for the case abs(k_cos_theta / k + 1) < eps
    # tricky to do with 3d tensors, but maybe it's ok without it...

    quat[:, :, 0:1] = k_cos_theta + k
    quat[:, :, 1:] = torch.cross(u, v, dim=-1)
    return normalize(quat), normalize(quat)[:, :, 0:1]#k_cos_theta + k

class quaternion_progress(torch.nn.Module):
    ''' An "unrolled" (i.e. no recursion) xyz to quat converter layer. Needs to be updated if the skeleton graph changes.'''

    def __init__(self):
        super(quaternion_progress, self).__init__()

    def forward(self, data):#b,f,21,3
# def quaternion_progress(data):
        result_root_positions = torch.zeros((data.shape[0], data.shape[1], 3)).to(device)
        result_quats = torch.zeros((data.shape[0], data.shape[1], 21 * 4)).to(device)
        result_theta = torch.zeros((data.shape[0], data.shape[1], 21 * 1)).to(device)
        result_lengths = torch.zeros((data.shape[0], data.shape[1], 21 * 1)).to(device)


        # Hand: Anchor0

        offset = torch.squeeze(data[:, :, [0]], dim = 2) # b,f,3
        result_root_positions[:, :, 0:3] = offset
        p0 = X_AXIS.repeat(data.shape[0], data.shape[1], 1).float().to(device)
        p1 = torch.zeros((data.shape[0], data.shape[1], 3)).float().to(device)

        # Hands T1: 
        p_Hands_T1 = torch.squeeze(data[:, :, [1]], dim = 2) - offset
        u = p0.float() - p1.float()
        v = p_Hands_T1.float() - p1.float()
        result_quats[:, :, 0:4], result_theta[:, :, 0:1] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 0:1] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands T2: 
        p_Hands_T2 = torch.squeeze(data[:, :, [2]], dim = 2) - offset
        u = p1.float() - p_Hands_T1.float()
        v = p_Hands_T2.float() - p_Hands_T1.float()
        result_quats[:, :, 4:8], result_theta[:, :, 1:2] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 1:2] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands T3: 
        p_Hands_T3 = torch.squeeze(data[:, :, [3]], dim = 2)- offset
        u = p_Hands_T1.float() - p_Hands_T2.float()
        v = p_Hands_T3.float() - p_Hands_T2.float()
        result_quats[:, :, 8:12], result_theta[:, :, 2:3] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 2:3] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands T4: 
        p_Hands_T4 = torch.squeeze(data[:, :, [4]], dim = 2) - offset
        u = p_Hands_T2.float() - p_Hands_T3.float()
        v = p_Hands_T4.float() - p_Hands_T3.float()
        result_quats[:, :, 12:16], result_theta[:, :, 3:4] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 3:4] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands I5: 
        p_Hands_I5 = torch.squeeze(data[:, :, [5]], dim = 2) - offset
        u = p0.float() - p1.float()
        v = p_Hands_I5.float() - p1.float()
        result_quats[:, :, 16:20], result_theta[:, :, 4:5] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 4:5] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands I6: 
        p_Hands_I6 = torch.squeeze(data[:, :, [6]], dim = 2) - offset
        u = p1.float() - p_Hands_I5.float()
        v = p_Hands_I6.float() - p_Hands_I5.float()
        result_quats[:, :, 20:24], result_theta[:, :, 5:6] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 5:6] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands I7: 
        p_Hands_I7 = torch.squeeze(data[:, :, [7]], dim = 2) - offset
        u = p_Hands_I5.float() - p_Hands_I6.float()
        v = p_Hands_I7.float() - p_Hands_I6.float()
        result_quats[:, :, 24:28], result_theta[:, :, 6:7] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 6:7] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands I8: 
        p_Hands_I8 = torch.squeeze(data[:, :, [8]], dim = 2) - offset
        u = p_Hands_I6.float() - p_Hands_I7.float()
        v = p_Hands_I8.float() - p_Hands_I7.float()
        result_quats[:, :, 28:32], result_theta[:, :, 7:8] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 7:8] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands M9: 
        p_Hands_M9 = torch.squeeze(data[:, :, [9]], dim = 2) - offset
        u = p1.float() - p_Hands_I5.float()
        v = p_Hands_M9.float() - p_Hands_I5.float()
        result_quats[:, :, 32:36], result_theta[:, :, 8:9] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 8:9] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands M10: 
        p_Hands_M10 = torch.squeeze(data[:, :, [10]], dim = 2) - offset
        u = p_Hands_I5.float() - p_Hands_M9.float()
        v = p_Hands_M10.float() - p_Hands_M9.float()
        result_quats[:, :, 36:40] , result_theta[:, :, 9:10]= get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 9:10] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands M11: 
        p_Hands_M11 = torch.squeeze(data[:, :, [11]], dim = 2) - offset
        u = p_Hands_M9.float() - p_Hands_M10.float()
        v = p_Hands_M11.float() - p_Hands_M10.float()
        result_quats[:, :, 40:44], result_theta[:, :, 10:11] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 10:11] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands M12: 
        p_Hands_M12 = torch.squeeze(data[:, :, [12]], dim = 2) - offset
        u = p_Hands_M10.float() - p_Hands_M11.float()
        v = p_Hands_M12.float() - p_Hands_M11.float()
        result_quats[:, :, 44:48], result_theta[:, :, 11:12] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 11:12] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands R13: 
        p_Hands_R13 = torch.squeeze(data[:, :, [13]], dim = 2) - offset
        u = p_Hands_I5.float() - p_Hands_M9.float()
        v = p_Hands_R13.float() - p_Hands_M9.float()
        result_quats[:, :, 48:52], result_theta[:, :, 12:13] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 12:13] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands R14: 
        p_Hands_R14 = torch.squeeze(data[:, :, [14]], dim = 2) - offset
        u = p_Hands_M9.float() - p_Hands_R13.float()
        v = p_Hands_R14.float() - p_Hands_R13.float()
        result_quats[:, :, 52:56], result_theta[:, :, 13:14] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 13:14] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands R15: 
        p_Hands_R15 = torch.squeeze(data[:, :, [15]], dim = 2) - offset
        u = p_Hands_R13.float() - p_Hands_R14.float()
        v = p_Hands_R15.float() - p_Hands_R14.float()
        result_quats[:, :, 56:60], result_theta[:, :, 14:15] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 14:15] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands R16: 
        p_Hands_R16 = torch.squeeze(data[:, :, [16]], dim = 2) - offset
        u = p_Hands_R14.float() - p_Hands_R15.float()
        v = p_Hands_R16.float() - p_Hands_R15.float()
        result_quats[:, :, 60:64], result_theta[:, :, 15:16] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 15:16] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands P17: 
        p_Hands_P17 = torch.squeeze(data[:, :, [17]], dim = 2) - offset
        u = p_Hands_M9.float() - p_Hands_R13.float()
        v = p_Hands_P17.float() - p_Hands_R13.float()
        result_quats[:, :, 64:68], result_theta[:, :, 16:17] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 16:17] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands P18: 
        p_Hands_P18 = torch.squeeze(data[:, :, [18]], dim = 2) - offset
        u = p_Hands_R13.float() - p_Hands_P17.float()
        v = p_Hands_P18.float() - p_Hands_P17.float()
        result_quats[:, :, 68:72], result_theta[:, :, 17:18] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 17:18] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands P19: 
        p_Hands_P19 = torch.squeeze(data[:, :, [19]], dim = 2) - offset
        u = p_Hands_P17.float() - p_Hands_P18.float()
        v = p_Hands_P19.float() - p_Hands_P18.float()
        result_quats[:, :, 72:76], result_theta[:, :, 18:19] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 18:19] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands P20: 
        p_Hands_P20 = torch.squeeze(data[:, :, [20]], dim = 2) - offset
        u = p_Hands_P18.float() - p_Hands_P19.float()
        v = p_Hands_P20.float() - p_Hands_P19.float()
        result_quats[:, :, 76:80], result_theta[:, :, 19:20] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 19:20] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Hands end: 
        p_Hands_END = torch.squeeze(data[:, :, [17]], dim = 2) - offset
        u = p0.float() - p1.float()
        v = p_Hands_END.float() - p1.float()
        result_quats[:, :, 80:84], result_theta[:, :, 20:21] = get_quat_rot_between(normalize(u), normalize(v))
        result_lengths[:, :, 20:21] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_L, T1: 
#         p_Hands_L_T1 = data[:, :, [0, 1, 2]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_L_T1.float() - p1.float()
#         result_quats[:, :, 84:88] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 21:22] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_L, T2: 
#         p_Hands_L_T2 = data[:, :, [3, 4, 5]] - offset
#         u = p1.float() - p_Hands_L_T1.float()
#         v = p_Hands_L_T2.float() - p_Hands_L_T1.float()
#         result_quats[:, :, 88:92] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 22:23] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_L, T3: 
#         p_Hands_L_T3 = data[:, :, [6, 7, 8]] - offset
#         u = p_Hands_L_T1.float() - p_Hands_L_T2.float()
#         v = p_Hands_L_T3.float() - p_Hands_L_T2.float()
#         result_quats[:, :, 92:96] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 23:24] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_L, T4: 
#         p_Hands_L_T4 = data[:, :, [9, 10, 11]] - offset
#         u = p_Hands_L_T2.float() - p_Hands_L_T3.float()
#         v = p_Hands_L_T4.float() - p_Hands_L_T3.float()
#         result_quats[:, :, 96:100] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 24:25] = torch.norm(v, p=None, dim=2, keepdim=True)


#         # Hand: Hands_R

#         offset = data[:, :, [150, 151, 152]]
#         result_root_positions[:, :, 3:6] = offset
#         p0 = X_AXIS.repeat(data.shape[0], data.shape[1], 1).float().to(device)
#         p1 = torch.zeros((data.shape[0], data.shape[1], 3)).float().to(device)

#         # Hands_R, Win: 
#         p_Hands_R_Win = data[:, :, [144, 145, 146]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_Win.float() - p1.float()
#         result_quats[:, :, 100:104] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 25:26] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, Ain: 
#         p_Hands_R_Ain = data[:, :, [138, 139, 140]] - offset
#         u = p1.float() - p_Hands_R_Win.float()
#         v = p_Hands_R_Ain.float() - p_Hands_R_Win.float()
#         result_quats[:, :, 104:108] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 26:27] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, Cout: 
#         p_Hands_R_Cout = data[:, :, [153, 154, 155]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_Cout.float() - p1.float()
#         result_quats[:, :, 108:112] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 27:28] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, Wout: 
#         p_Hands_R_Wout = data[:, :, [147, 148, 149]] - offset
#         u = p1.float() - p_Hands_R_Cout.float()
#         v = p_Hands_R_Wout.float() - p_Hands_R_Cout.float()
#         result_quats[:, :, 112:116] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 28:29] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, Aout: 
#         p_Hands_R_Aout = data[:, :, [141, 142, 143]] - offset
#         u = p_Hands_R_Cout.float() - p_Hands_R_Wout.float()
#         v = p_Hands_R_Aout.float() - p_Hands_R_Wout.float()
#         result_quats[:, :, 116:120] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 29:30] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, L1: 
#         p_Hands_R_L1 = data[:, :, [108, 109, 110]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_L1.float() - p1.float()
#         result_quats[:, :, 120:124] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 30:31] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, L2: 
#         p_Hands_R_L2 = data[:, :, [111, 112, 113]] - offset
#         u = p1.float() - p_Hands_R_L1.float()
#         v = p_Hands_R_L2.float() - p_Hands_R_L1.float()
#         result_quats[:, :, 124:128] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 31:32] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, L3: 
#         p_Hands_R_L3 = data[:, :, [114, 115, 116]] - offset
#         u = p_Hands_R_L1.float() - p_Hands_R_L2.float()
#         v = p_Hands_R_L3.float() - p_Hands_R_L2.float()
#         result_quats[:, :, 128:132] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 32:33] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, L4: 
#         p_Hands_R_L4 = data[:, :, [117, 118, 119]] - offset
#         u = p_Hands_R_L2.float() - p_Hands_R_L3.float()
#         v = p_Hands_R_L4.float() - p_Hands_R_L3.float()
#         result_quats[:, :, 132:136] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 33:34] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, R1: 
#         p_Hands_R_R1 = data[:, :, [96, 97, 98]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_R1.float() - p1.float()
#         result_quats[:, :, 136:140] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 34:35] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, R2: 
#         p_Hands_R_R2 = data[:, :, [99, 100, 101]] - offset
#         u = p1.float() - p_Hands_R_R1.float()
#         v = p_Hands_R_R2.float() - p_Hands_R_R1.float()
#         result_quats[:, :, 140:144] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 35:36] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, R3: 
#         p_Hands_R_R3 = data[:, :, [102, 103, 104]] - offset
#         u = p_Hands_R_R1.float() - p_Hands_R_R2.float()
#         v = p_Hands_R_R3.float() - p_Hands_R_R2.float()
#         result_quats[:, :, 144:148] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 36:37] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, R4: 
#         p_Hands_R_R4 = data[:, :, [105, 106, 107]] - offset
#         u = p_Hands_R_R2.float() - p_Hands_R_R3.float()
#         v = p_Hands_R_R4.float() - p_Hands_R_R3.float()
#         result_quats[:, :, 148:152] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 37:38] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, M1: 
#         p_Hands_R_M1 = data[:, :, [84, 85, 86]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_M1.float() - p1.float()
#         result_quats[:, :, 152:156] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 38:39] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, M2: 
#         p_Hands_R_M2 = data[:, :, [87, 88, 89]] - offset
#         u = p1.float() - p_Hands_R_M1.float()
#         v = p_Hands_R_M2.float() - p_Hands_R_M1.float()
#         result_quats[:, :, 156:160] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 39:40] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, M3: 
#         p_Hands_R_M3 = data[:, :, [90, 91, 92]] - offset
#         u = p_Hands_R_M1.float() - p_Hands_R_M2.float()
#         v = p_Hands_R_M3.float() - p_Hands_R_M2.float()
#         result_quats[:, :, 160:164] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 40:41] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, M4: 
#         p_Hands_R_M4 = data[:, :, [93, 94, 95]] - offset
#         u = p_Hands_R_M2.float() - p_Hands_R_M3.float()
#         v = p_Hands_R_M4.float() - p_Hands_R_M3.float()
#         result_quats[:, :, 164:168] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 41:42] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, I1: 
#         p_Hands_R_I1 = data[:, :, [72, 73, 74]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_I1.float() - p1.float()
#         result_quats[:, :, 168:172] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 42:43] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, I2: 
#         p_Hands_R_I2 = data[:, :, [75, 76, 77]] - offset
#         u = p1.float() - p_Hands_R_I1.float()
#         v = p_Hands_R_I2.float() - p_Hands_R_I1.float()
#         result_quats[:, :, 172:176] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 43:44] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, I3: 
#         p_Hands_R_I3 = data[:, :, [78, 79, 80]] - offset
#         u = p_Hands_R_I1.float() - p_Hands_R_I2.float()
#         v = p_Hands_R_I3.float() - p_Hands_R_I2.float()
#         result_quats[:, :, 176:180] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 44:45] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, I4: 
#         p_Hands_R_I4 = data[:, :, [81, 82, 83]] - offset
#         u = p_Hands_R_I2.float() - p_Hands_R_I3.float()
#         v = p_Hands_R_I4.float() - p_Hands_R_I3.float()
#         result_quats[:, :, 180:184] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 45:46] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, T1: 
#         p_Hands_R_T1 = data[:, :, [60, 61, 62]] - offset
#         u = p0.float() - p1.float()
#         v = p_Hands_R_T1.float() - p1.float()
#         result_quats[:, :, 184:188] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 46:47] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, T2: 
#         p_Hands_R_T2 = data[:, :, [63, 64, 65]] - offset
#         u = p1.float() - p_Hands_R_T1.float()
#         v = p_Hands_R_T2.float() - p_Hands_R_T1.float()
#         result_quats[:, :, 188:192] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 47:48] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, T3: 
#         p_Hands_R_T3 = data[:, :, [66, 67, 68]] - offset
#         u = p_Hands_R_T1.float() - p_Hands_R_T2.float()
#         v = p_Hands_R_T3.float() - p_Hands_R_T2.float()
#         result_quats[:, :, 192:196] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 48:49] = torch.norm(v, p=None, dim=2, keepdim=True)

#         # Hands_R, T4: 
#         p_Hands_R_T4 = data[:, :, [69, 70, 71]] - offset
#         u = p_Hands_R_T2.float() - p_Hands_R_T3.float()
#         v = p_Hands_R_T4.float() - p_Hands_R_T3.float()
#         result_quats[:, :, 196:200] = get_quat_rot_between(normalize(u), normalize(v))
#         result_lengths[:, :, 49:50] = torch.norm(v, p=None, dim=2, keepdim=True)

        # Return result:
        return result_root_positions, result_quats, result_theta, result_lengths # b,f,3  b,f,84  b,f,21  b,f,21
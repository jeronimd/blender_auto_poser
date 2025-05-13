import torch

# # batch*n
# def normalize_vector(v, return_mag=False, eps: float = 1e-8):
#     batch = v.shape[0]
#     v_mag = torch.sqrt(v.pow(2).sum(1))
#     v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([eps]).type_as(v)))
#     v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
#     v = v / v_mag
#     if return_mag:
#         return v, v_mag[:, 0]
#     else:
#         return v


# # u, v batch*n
# def cross_product(u, v):
#     batch = u.shape[0]

#     i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
#     j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
#     k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

#     out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

#     return out


# # q of size (N, 4)
# # quaternion order is w, x, y, z
# # output of size (N, 3, 3)
# def convert_quat_to_matrix_unit(q):
#     batch = q.shape[0]

#     qw = q[..., 0].contiguous().view(batch, 1)
#     qx = q[..., 1].contiguous().view(batch, 1)
#     qy = q[..., 2].contiguous().view(batch, 1)
#     qz = q[..., 3].contiguous().view(batch, 1)

#     # Unit quaternion rotation matrices computation
#     xx = qx * qx
#     yy = qy * qy
#     zz = qz * qz
#     xy = qx * qy
#     xz = qx * qz
#     yz = qy * qz
#     xw = qx * qw
#     yw = qy * qw
#     zw = qz * qw

#     row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
#     row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
#     row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

#     matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

#     return matrix


# # q of size (N, 4)
# # quaternion order is w, x, y, z
# # output of size (N, 3, 3)
# def convert_quat_to_matrix(q, eps: float = 1e-8):
#     return convert_quat_to_matrix_unit(normalize_vector(q, eps=eps).contiguous())


# def convert_matrix_to_ortho(mat):
#     mat = mat.reshape([-1, 3, 3])
#     r6d = torch.cat([mat[..., 0], mat[..., 1]], dim=-1)
#     return r6d


# def convert_ortho_to_matrix(ortho6d, eps: float = 1e-8):
#     x_raw = ortho6d[:, 0:3]  # batch*3
#     y_raw = ortho6d[:, 3:6]  # batch*3

#     x = normalize_vector(x_raw, eps=eps)  # batch*3
#     z = cross_product(x, y_raw)  # batch*3
#     z = normalize_vector(z, eps=eps)  # batch*3
#     y = cross_product(z, x)  # batch*3

#     x = x.view(-1, 3, 1)
#     y = y.view(-1, 3, 1)
#     z = z.view(-1, 3, 1)
#     matrix = torch.cat((x, y, z), 2)  # batch*3*3
#     return matrix


# euler batch*4
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def convert_euler_to_matrix(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


# # matrices batch*3*3
# # both matrix are orthogonal rotation matrices
# # out theta between 0 to 180 degree batch
# def compute_geodesic_distance_from_two_matrices(m1, m2):
#     batch = m1.shape[0]
#     m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

#     cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
#     cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)))
#     cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)) * -1)

#     theta = torch.acos(cos)

#     # theta = torch.min(theta, 2*np.pi - theta)

#     return theta


##################################################################################################################
##################################################################################################################


def normalize_vector(v, return_mag=False, eps: float = 1e-5):
    batch = v.shape[0]
    # Square and sum for magnitude calculation
    v_squared = v.pow(2)
    v_sum = v_squared.sum(1, keepdim=True)

    # Create epsilon tensor directly on the same device
    eps_tensor = torch.full_like(v_sum, eps)

    # Use maximum for stability, avoid Variable constructor
    v_mag = torch.sqrt(torch.maximum(v_sum, eps_tensor))

    # Expand for division
    v_mag_expanded = v_mag.expand(batch, v.shape[1])

    # Normalize
    v_normalized = v / v_mag_expanded

    if return_mag:
        return v_normalized, v_mag.view(batch)
    else:
        return v_normalized


def convert_ortho_to_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    a1 = ortho6d[..., :3]
    a2 = ortho6d[..., 3:]

    # Normalize first axis with epsilon
    norm_a1 = torch.norm(a1, dim=-1, keepdim=True)
    a1 = a1 / torch.clamp(norm_a1, min=eps)

    # Orthogonalize second axis
    a2 = a2 - torch.sum(a2 * a1, dim=-1, keepdim=True) * a1
    norm_a2 = torch.norm(a2, dim=-1, keepdim=True)
    a2 = a2 / torch.clamp(norm_a2, min=eps)

    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack((a1, a2, a3), dim=-1)


def convert_ortho_to_quat(ortho6d: torch.Tensor) -> torch.Tensor:
    matrix = convert_ortho_to_matrix(ortho6d)
    return convert_matrix_to_quat(matrix)


def convert_quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    # Add numerical stability fixes
    eps = 1e-8

    # 1. Normalize input with epsilon protection
    quaternion = quaternion.clone()  # Avoid modifying original tensor
    norm = torch.norm(quaternion, dim=-1, keepdim=True)
    quaternion = torch.where(norm > eps,
                             quaternion / torch.clamp(norm, min=eps),
                             torch.tensor([1.0, 0.0, 0.0, 0.0], device=quaternion.device))

    # Ensure consistent handedness (w > 0)
    quaternion = torch.where(quaternion[..., 0:1] < 0, -quaternion, quaternion)

    # 2. Split components with stability
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    # 3. Compute components with clamping
    xx = torch.clamp(x * x, -1.0, 1.0)
    yy = torch.clamp(y * y, -1.0, 1.0)
    zz = torch.clamp(z * z, -1.0, 1.0)
    xy = torch.clamp(x * y, -1.0, 1.0)
    xz = torch.clamp(x * z, -1.0, 1.0)
    yz = torch.clamp(y * z, -1.0, 1.0)
    wx = torch.clamp(w * x, -1.0, 1.0)
    wy = torch.clamp(w * y, -1.0, 1.0)
    wz = torch.clamp(w * z, -1.0, 1.0)

    # 4. Build matrix with safe operations
    row0 = torch.stack((
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy)
    ), dim=-1)

    row1 = torch.stack((
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx)
    ), dim=-1)

    row2 = torch.stack((
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (xx + yy)
    ), dim=-1)

    # 5. Final validation
    matrix = torch.stack((row0, row1, row2), dim=-2)
    matrix = torch.where(torch.isfinite(matrix), matrix, torch.eye(3, device=matrix.device))

    return matrix


def convert_quat_to_ortho(quat: torch.Tensor) -> torch.Tensor:
    matrix = convert_quat_to_matrix(quat)
    return torch.cat((matrix[..., 0], matrix[..., 1]), dim=-1)


def convert_matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")

    batch_dim = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    # Compute trace for case identification
    eps = 1e-8
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    # Initialize quaternion tensor
    q = torch.zeros(m.shape[0], 4, device=matrix.device)

    # Case 1: Trace > 0
    mask_trace_positive = trace > 0
    if mask_trace_positive.any():
        # Using max to ensure sqrt argument is positive
        S = torch.sqrt(torch.clamp(trace[mask_trace_positive] + 1.0, min=eps)) * 2

        q[mask_trace_positive, 0] = 0.25 * S
        q[mask_trace_positive, 1] = (m[mask_trace_positive, 2, 1] - m[mask_trace_positive, 1, 2]) / S
        q[mask_trace_positive, 2] = (m[mask_trace_positive, 0, 2] - m[mask_trace_positive, 2, 0]) / S
        q[mask_trace_positive, 3] = (m[mask_trace_positive, 1, 0] - m[mask_trace_positive, 0, 1]) / S

    # Case 2: Trace <= 0
    remaining_indices = torch.nonzero(~mask_trace_positive).squeeze(-1)
    if len(remaining_indices) > 0:
        m_remaining = m[remaining_indices]

        # Find the largest diagonal element to determine the dominant component
        diag = torch.stack([m_remaining[:, 0, 0], m_remaining[:, 1, 1], m_remaining[:, 2, 2]], dim=-1)
        max_diag_indices = torch.argmax(diag, dim=-1)

        for i in range(3):
            mask_max_i = max_diag_indices == i
            if not mask_max_i.any():
                continue

            indices = remaining_indices[mask_max_i]
            m_sub = m_remaining[mask_max_i]

            next_i = (i + 1) % 3
            next_next_i = (i + 2) % 3

            S = torch.sqrt(torch.clamp(
                1.0 + m_sub[:, i, i] - m_sub[:, next_i, next_i] - m_sub[:, next_next_i, next_next_i],
                min=eps
            )) * 2

            q[indices, i + 1] = 0.25 * S
            q[indices, 0] = (m_sub[:, next_next_i, next_i] - m_sub[:, next_i, next_next_i]) / S
            q[indices, next_i + 1] = (m_sub[:, next_i, i] + m_sub[:, i, next_i]) / S
            q[indices, next_next_i + 1] = (m_sub[:, next_next_i, i] + m_sub[:, i, next_next_i]) / S

    # Normalize quaternions
    q_norm = torch.norm(q, dim=1, keepdim=True)
    q = q / torch.clamp(q_norm, min=eps)

    # Ensure consistent handedness (w >= 0)
    q = torch.where(q[:, 0:1] < 0, -q, q)

    return q.reshape(*batch_dim, 4)


def convert_matrix_to_ortho(mat) -> torch.Tensor:
    mat = mat.reshape([-1, 3, 3])
    r6d = torch.cat([mat[..., 0], mat[..., 1]], dim=-1)
    return r6d


# I prefer the old version even if it is deprecated, since it had a better result
def compute_geodesic_distance_from_two_matrices(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    # Compute rotation matrix representing the relative rotation
    m = torch.bmm(m1, m2.transpose(1, 2))

    # Compute the angle from the trace
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # Handle numerical stability (modernized version without deprecated Variable)
    eps = 1e-6
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)

    # Compute the angle
    theta = torch.acos(cos)
    return theta

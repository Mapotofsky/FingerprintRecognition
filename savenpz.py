import numpy as np

from utils import *
import os


def save_npz(path,img):
    num = [0]
    fingerprint = img

    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)

    gx2, gy2 = gx ** 2, gy ** 2
    gm = np.sqrt(gx2 + gy2)

    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize=False)

    thr = sum_gm.max() * 0.2
    mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)

    W = (23, 23)
    gxx = cv.boxFilter(gx2, -1, W, normalize=False)
    gyy = cv.boxFilter(gy2, -1, W, normalize=False)
    gxy = cv.boxFilter(gx * gy, -1, W, normalize=False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv.sqrt((gxx_gyy ** 2 + gxy2 ** 2)), sum_gxx_gyy, out=np.zeros_like(gxx),
                          where=sum_gxx_gyy != 0)

    region = fingerprint[10:90, 80:130]
    show(region, path='img\\region', num=num)

    smoothed = cv.blur(region, (5, 5), -1)
    xs = np.sum(smoothed, 1)

    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    distances = local_maxima[1:] - local_maxima[:-1]

    ridge_period = np.average(distances)

    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]

    nf = 255 - fingerprint
    all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
    y_coords, x_coords = np.indices(fingerprint.shape)
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)

    _, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)

    skeleton = cv.ximgproc.thinning(ridge_lines, thinningType=cv.ximgproc.THINNING_GUOHALL)

    def compute_crossing_number(values):
        return np.count_nonzero(values < np.roll(values, -1))

    cn_filter = np.array([[1, 2, 4],
                          [128, 0, 8],
                          [64, 32, 16]
                          ])

    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

    skeleton01 = np.where(skeleton != 0, 1, 0).astype(np.uint8)
    cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType=cv.BORDER_CONSTANT)
    cn = cv.LUT(cn_values, cn_lut)
    cn[skeleton == 0] = 0

    minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]

    mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,
                    1:-1]

    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > 10, minutiae))

    def compute_next_ridge_following_directions(previous_direction, values):
        next_positions = np.argwhere(values != 0).ravel().tolist()
        if len(next_positions) > 0 and previous_direction != 8:
            next_positions.sort(key=lambda d: 4 - abs(abs(d - previous_direction) - 4))
            if next_positions[-1] == (
                    previous_direction + 4) % 8:
                next_positions = next_positions[:-1]
        return next_positions

    r2 = 2 ** 0.5
    xy_steps = [(-1, -1, r2), (0, -1, 1), (1, -1, r2), (1, 0, 1), (1, 1, r2), (0, 1, 1), (-1, 1, r2), (-1, 0, 1)]
    nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

    def follow_ridge_and_compute_angle(x, y, d=8):
        px, py = x, y
        length = 0.0
        while length < 20:
            next_directions = nd_lut[cn_values[py, px]][d]
            if len(next_directions) == 0:
                break
            if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
                break
            d = next_directions[0]
            ox, oy, l = xy_steps[d]
            px += ox
            py += oy
            length += l
        return math.atan2(-py + y, px - x) if length >= 10 else None

    valid_minutiae = []
    for x, y, term in filtered_minutiae:
        d = None
        if term:
            d = follow_ridge_and_compute_angle(x, y)
        else:
            dirs = nd_lut[cn_values[y, x]][8]
            if len(dirs) == 3:
                angles = [follow_ridge_and_compute_angle(x + xy_steps[d][0], y + xy_steps[d][1], d) for d in dirs]
                if all(a is not None for a in angles):
                    a1, a2 = min(((angles[i], angles[(i + 1) % 3]) for i in range(3)),
                                 key=lambda t: angle_abs_difference(t[0], t[1]))
                    d = angle_mean(a1, a2)
        if d is not None:
            valid_minutiae.append((x, y, term, d))

    mcc_radius = 70
    mcc_size = 16

    g = 2 * mcc_radius / mcc_size
    x = np.arange(mcc_size) * g - (mcc_size / 2) * g + g / 2
    y = x[..., np.newaxis]
    iy, ix = np.nonzero(x ** 2 + y ** 2 <= mcc_radius ** 2)
    ref_cell_coords = np.column_stack((x[ix], x[iy]))

    mcc_sigma_s = 7.0
    mcc_tau_psi = 400.0
    mcc_mu_psi = 1e-2

    def Gs(t_sqr):
        """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
        return np.exp(-0.5 * t_sqr / (mcc_sigma_s ** 2)) / (math.tau ** 0.5 * mcc_sigma_s)

    def Psi(v):
        """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
        return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))

    xyd = np.array(
        [(x, y, d) for x, y, _, d in valid_minutiae])

    # rot: n x 2 x 2
    d_cos, d_sin = np.cos(xyd[:, 2]).reshape((-1, 1, 1)), np.sin(xyd[:, 2]).reshape((-1, 1, 1))
    rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

    # rot@ref_cell_coords.T : n x 2 x c
    # xy : n x 2
    xy = xyd[:, :2]
    # cell_coords: n x c x 2
    cell_coords = np.transpose(rot @ ref_cell_coords.T + xy[:, :, np.newaxis], [0, 2, 1])

    dists = np.sum((cell_coords[:, :, np.newaxis, :] - xy) ** 2, -1)

    # cs : n x c x n
    cs = Gs(dists)
    diag_indices = np.arange(cs.shape[0])
    cs[diag_indices, :, diag_indices] = 0

    # local_structures : n x c
    local_structures = Psi(np.sum(cs, -1))
    f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
    np.savez(path, m1, ls1)
    print('save is ok')


if __name__ == '__main__':
    img= cv.imread('pic/origin.png', cv.IMREAD_GRAYSCALE)
    save_npz('pic/origin.npz',img)
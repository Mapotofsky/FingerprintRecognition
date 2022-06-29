import numpy as np

from utils import *
import os
def get_all_imgs(path):
    import matplotlib.pyplot as plt
    from ipywidgets import interact

    if not os.path.exists('img'):
        os.mkdir('img')
    num=[0]

    fingerprint = cv.imread(path, cv.IMREAD_GRAYSCALE)
    show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}',path='img\\ori',num=num)

    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
    show((gx, 'Gx'), (gy, 'Gy'),path='img\\sobel',num=num)

    gx2, gy2 = gx**2, gy**2
    gm = np.sqrt(gx2 + gy2)
    l=show((gx2, 'Gx**2'), (gy2, 'Gy**2'), (gm, 'Gradient magnitude'),path='img\\sobel_squared',num=num)

    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
    show(sum_gm, 'Integral of the gradient magnitude',path='img\\integral',num=num)

    thr = sum_gm.max() * 0.2
    mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    show(fingerprint, mask, cv.merge((mask, fingerprint, fingerprint)),path='img\\threshold',num=num)
    W = (23, 23)
    gxx = cv.boxFilter(gx2, -1, W, normalize = False)
    gyy = cv.boxFilter(gy2, -1, W, normalize = False)
    gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
    show(draw_orientations(fingerprint, orientations, strengths, mask, 1, 16), 'Orientation image',path='img\\Orientation',num=num)

    region = fingerprint[10:90,80:130]  # 截取局部指纹计算频率，这是个超参数
    show(region,path='img\\region',num=num)

    smoothed = cv.blur(region, (5,5), -1)
    xs = np.sum(smoothed, 1)

    x = np.arange(region.shape[0])
    f, axarr = plt.subplots(1,2, sharey = True)
    axarr[0].imshow(region,cmap='gray')
    axarr[1].plot(xs, x)
    axarr[1].set_ylim(region.shape[0]-1,0)
    if not os.path.exists('img\\plot1'):
        os.mkdir('img\\plot1')
    plt.savefig("img\\plot1\\plot1.png")
    plt.clf()

    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    x = np.arange(region.shape[0])
    plt.plot(x, xs)
    plt.xticks(local_maxima)
    plt.grid(True, axis='x')
    if not os.path.exists('img\\plot2'):
        os.mkdir('img\\plot2')
    plt.savefig("img\\plot2\\plot2.png")

    distances = local_maxima[1:] - local_maxima[:-1]

    ridge_period = np.average(distances)

    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

    show(*gabor_bank,path='img\\8_filters',num=num)

    nf = 255-fingerprint
    all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
    show(nf, *all_filtered,path='img\\after_filters',num=num)


    y_coords, x_coords = np.indices(fingerprint.shape)
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
    show(fingerprint, enhanced,path='img\\enhanced',num=num)

    _, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
    show(fingerprint, ridge_lines, cv.merge((ridge_lines, fingerprint, fingerprint)),path='img\\binarization',num=num)

    skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
    show(skeleton, cv.merge((fingerprint, fingerprint, skeleton)),path='img\\thinning',num=num)

    def compute_crossing_number(values):
        return np.count_nonzero(values < np.roll(values, -1))

    cn_filter = np.array([[  1,  2,  4],
                          [128,  0,  8],
                          [ 64, 32, 16]
                         ])

    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

    skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
    cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
    cn = cv.LUT(cn_values, cn_lut)
    cn[skeleton==0] = 0

    minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]

    show(draw_minutiae(fingerprint, minutiae), skeleton, draw_minutiae(skeleton, minutiae),path='img\\crossing',num=num)


    mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
    show(mask, mask_distance,path='img\\mask',num=num)


    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))

    show(draw_minutiae(fingerprint, filtered_minutiae), skeleton, draw_minutiae(skeleton, filtered_minutiae),path='img\\minutiae',num=num)



    def compute_next_ridge_following_directions(previous_direction, values):
        next_positions = np.argwhere(values!=0).ravel().tolist()
        if len(next_positions) > 0 and previous_direction != 8:
            next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
            if next_positions[-1] == (previous_direction + 4) % 8:
                next_positions = next_positions[:-1]
        return next_positions

    r2 = 2**0.5

    xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

    nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

    def follow_ridge_and_compute_angle(x, y, d = 8):
        px, py = x, y
        length = 0.0
        while length < 20:
            next_directions = nd_lut[cn_values[py,px]][d]
            if len(next_directions) == 0:
                break
            if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
                break
            d = next_directions[0]
            ox, oy, l = xy_steps[d]
            px += ox ; py += oy ; length += l
        return math.atan2(-py+y, px-x) if length >= 10 else None

    valid_minutiae = []
    for x, y, term in filtered_minutiae:
        d = None
        if term:
            d = follow_ridge_and_compute_angle(x, y)
        else:
            dirs = nd_lut[cn_values[y,x]][8]
            if len(dirs)==3:
                angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
                if all(a is not None for a in angles):
                    a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                    d = angle_mean(a1, a2)
        if d is not None:
            valid_minutiae.append( (x, y, term, d) )

    show(draw_minutiae(fingerprint, valid_minutiae),path='img\\draw_minutiae',num=num)


    mcc_radius = 70
    mcc_size = 16

    g = 2 * mcc_radius / mcc_size
    x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
    y = x[..., np.newaxis]
    iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
    ref_cell_coords = np.column_stack((x[ix], x[iy]))

    mcc_sigma_s = 7.0
    mcc_tau_psi = 400.0
    mcc_mu_psi = 1e-2

    def Gs(t_sqr):
        """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
        return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

    def Psi(v):
        """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
        return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))


    xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae])

    # rot: n x 2 x 2
    d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
    rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

    # rot@ref_cell_coords.T : n x 2 x c
    # xy : n x 2
    xy = xyd[:,:2]
    # cell_coords: n x c x 2
    cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

    # cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
    # xy                                 : (1 x 1) x n x 2
    # cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
    # dists: n x c x n
    dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

    # cs : n x c x n
    cs = Gs(dists)
    diag_indices = np.arange(cs.shape[0])
    cs[diag_indices,:,diag_indices] = 0

    # local_structures : n x c
    local_structures = Psi(np.sum(cs, -1))
    for i in range(0,len(valid_minutiae)-1):
        show(draw_minutiae_and_cylinder(fingerprint, ref_cell_coords, valid_minutiae, local_structures, i),path='img\\xijiedian1',num=num)

    f1, m1, ls1 = fingerprint, valid_minutiae, local_structures

    files = os.listdir('pic/samples')  # 指纹库
    res_pairs = []
    res_score = 0
    for i in range(len(files)//2):
        ofn = f'pic/samples/{i}'
        f2, (mm2, ls2) = cv.imread(f'{ofn}.png', cv.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()
        m2 = [[0 for _ in range(len(mm2[0]))] for _ in range(len(mm2))]
        for i in range(len(mm2)):
            m2[i][0] = mm2[i][0].astype(int)
            m2[i][1] = mm2[i][1].astype(int)
            m2[i][2] = mm2[i][2].astype(bool)
            m2[i][3] = mm2[i][3]
        # ls1                       : n1 x  c
        # ls1[:,np.newaxis,:]       : n1 x  1 x c
        # ls2                       : (1 x) n2 x c
        # ls1[:,np.newaxis,:] - ls2 : n1 x  n2 x c
        # dists                     : n1 x  n2
        dists = np.sqrt(np.sum((ls1[:,np.newaxis,:] - ls2)**2, -1))
        dists /= (np.sqrt(np.sum(ls1**2, 1))[:,np.newaxis] + np.sqrt(np.sum(ls2**2, 1)))

        num_p = 5
        pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
        score = 1 - np.mean(dists[pairs[0], pairs[1]])

        if (res_score < score):
            res_pairs = pairs
            res_score = score
    
    for i in range(0, len(res_pairs[0]) - 1):
        show(draw_match_pairs(f1, m1, ls1, f2, m2, ls2, ref_cell_coords, res_pairs, i, True),path='img\\xijiedian2',num=num)
    
    with open("img/p.txt","w") as f:
        f.write(f"{res_score}")
    
    return res_score

if __name__=='__main__':
    get_all_imgs()
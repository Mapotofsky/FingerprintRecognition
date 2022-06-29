import math
import numpy as np
import cv2 as cv
import urllib.request
import base64
import os

def show(*images, enlarge_small_images = True, max_per_row = -1, font_size = 0,num,path):
    if not os.path.exists(path):
        os.mkdir(path)
    if len(images) == 2 and type(images[1])==str:
        images = [(images[0], images[1])]

    def convert_for_display(img):
        if img.dtype!=np.uint8:
            a, b = img.min(), img.max()
            if a==b:
                offset, mult, d = 0, 0, 1
            elif a<0:
                offset, mult, d = 128, 127, max(abs(a), abs(b))
            else:
                offset, mult, d = 0, 255, b
            img = np.clip(offset + mult*(img.astype(float))/d, 0, 255).astype(np.uint8)
        return img

    def convert(imgOrTuple):
        try:
            img, title = imgOrTuple
            if type(title)!=str:
                img, title = imgOrTuple, ''
        except ValueError:
            img, title = imgOrTuple, ''
        if type(img)==str:
            data = img
        else:
            img = convert_for_display(img)
            if enlarge_small_images:
                REF_SCALE = 100
                h, w = img.shape[:2]
                if h<REF_SCALE or w<REF_SCALE:
                    scale = max(1, min(REF_SCALE//h, REF_SCALE//w))
                    img = cv.resize(img,(w*scale,h*scale), interpolation=cv.INTER_NEAREST)
            data = 'data:image/png;base64,' + base64.b64encode(cv.imencode('.png', img)[1]).decode('utf8')
        return img

    if max_per_row == -1:
        max_per_row = len(images)
    rows = [images[x:x+max_per_row] for x in range(0, len(images), max_per_row)]
    l=[]
    for r in rows:
        l = [convert(t) for t in r]
    for i in l:
        cv.imwrite(path+'\\'+str(num[0])+'.png',i)
        num[0]+=1

def load_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv.imdecode(image, cv.IMREAD_GRAYSCALE)

def draw_orientations(fingerprint, orientations, strengths, mask, scale = 3, step = 8, border = 0):
    if strengths is None:
        strengths = np.ones_like(orientations)
    h, w = fingerprint.shape
    sf = cv.resize(fingerprint, (w*scale, h*scale), interpolation = cv.INTER_NEAREST)
    res = cv.cvtColor(sf, cv.COLOR_GRAY2BGR)
    d = (scale // 2) + 1
    sd = (step+1)//2
    c = np.round(np.cos(orientations) * strengths * d * sd).astype(int)
    s = np.round(-np.sin(orientations) * strengths * d * sd).astype(int)
    thickness = 1 + scale // 5
    for y in range(border, h-border, step):
        for x in range(border, w-border, step):
            if mask is None or mask[y, x] != 0:
                ox, oy = c[y, x], s[y, x]
                cv.line(res, (d+x*scale-ox,d+y*scale-oy), (d+x*scale+ox,d+y*scale+oy), (255,0,0), thickness, cv.LINE_AA)
    return res

def draw_minutiae(fingerprint, minutiae, termination_color = (255,0,0), bifurcation_color = (0,0,255)):
    res = cv.cvtColor(fingerprint, cv.COLOR_GRAY2BGR)

    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d)==0:
            cv.drawMarker(res, (x,y), color, cv.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv.circle(res, (x,y), 3, color, 1, cv.LINE_AA)
            cv.line(res, (x,y), (x+ox,y-oy), color, 1, cv.LINE_AA)
    return res


_sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)

def gabor_kernel(period, orientation):
    f = cv.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    return f


def angle_abs_difference(a, b):
    return math.pi - abs(abs(a - b) - math.pi)

def angle_mean(a, b):
    return math.atan2((math.sin(a)+math.sin(b))/2, ((math.cos(a)+math.cos(b))/2))

# Utility functions for MCC
def draw_minutiae_and_cylinder(fingerprint, origin_cell_coords, minutiae, values, i, show_cylinder = True):

    def _compute_actual_cylinder_coordinates(x, y, t, d):
        c, s = math.cos(d), math.sin(d)
        rot = np.array([[c, s],[-s, c]])
        return (rot@origin_cell_coords.T + np.array([x,y])[:,np.newaxis]).T

    res = draw_minutiae(fingerprint, minutiae)
    if show_cylinder:
        for v, (cx, cy) in zip(values[i], _compute_actual_cylinder_coordinates(*minutiae[i])):
            cv.circle(res, (int(round(cx)), int(round(cy))), 3, (0,int(round(v*255)),0), 1, cv.LINE_AA)
    return res

def draw_match_pairs(f1, m1, v1, f2, m2, v2, cells_coords, pairs, i, show_cylinders = True):
    h1, w1 = f1.shape
    h2, w2 = f2.shape
    p1, p2 = pairs
    res = np.full((max(h1,h2), w1+w2, 3), 255, np.uint8)
    res[:h1,:w1] = draw_minutiae_and_cylinder(f1, cells_coords, m1, v1, p1[i], show_cylinders)
    res[:h2,w1:w1+w2] = draw_minutiae_and_cylinder(f2, cells_coords, m2, v2, p2[i], show_cylinders)
    for k, (i1, i2) in enumerate(zip(p1, p2)):
        (x1, y1, *_), (x2, y2, *_) = m1[i1], m2[i2]
        cv.line(res, (int(x1), int(y1)), (w1+int(x2), int(y2)), (0,0,255) if k!=i else (0,255,255), 1, cv.LINE_AA)
    return res
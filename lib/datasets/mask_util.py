import util
def draw_ann(mask, ann, color):
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        cnt = util.img.points_to_contours(poly);
        util.img.draw_contours(mask, cnt, idx = -1, color = color, border_width = -1)


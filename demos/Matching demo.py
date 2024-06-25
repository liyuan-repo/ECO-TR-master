import time
import matplotlib.pyplot as plt
from thop import profile
from dotmap import DotMap
from demos import demo_utils, noise_image
from sklearn.metrics import mean_squared_error
from src.config.default import get_cfg_defaults
from src.models.ecotr_engines import *
from src.models.utils.utils import *
from src.utils.misc import lower_config

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    dict = DotMap(lower_config(get_cfg_defaults()))
    dict.ecotr.engine.device = device
    fix_randomness(42)
    engine = ECOTR_Engine(dict.ecotr)
    torch.cuda.empty_cache()

    idx = 1
    engine.MAX_KPTS_NUM = 10000
    engine.ASPECT_RATIOS = [1.0, 1.5]

    img0_path = "../assets/4SARSets/pair4-1.png"
    img1_path = "../assets/4SARSets/pair4-2.png"
    output_path = "../output/fig/pair04.jpg"
    # img00 = noise_image.Additive_noise(img0_path, -2)
    # img0 = noise_image.Additive_noise(img0_path, 2)
    # output_path = "../output/4SARSets/pair4+0p05S.jpg"
    # img00 = noise_image.stripe_noise(img0_path, 0.12)
    # img0 = noise_image.stripe_noise(img0_path, 0.10)

    img0 = cv2.imread(img0_path)
    img0 = demo_utils.resize(img0, 512)

    img1 = cv2.imread(img1_path)
    img1 = demo_utils.resize(img1, 512)

    tic = time.time()
    # coarse2fine
    with torch.no_grad():

        matches = engine.forward(img0, img1, cycle=True, level='fine')

        mask = matches[:, -1] < 1e-2
        matches = matches[mask]

    canvas = draw_matches(img0, img1, matches.copy(), 'dot')

    kp_a, kp_b = matches[:, :2].copy(), matches[:, 2:4].copy()
    # canvas = draw_matches(img0, img1, kp_a, kp_b, matches.copy(), 'line')

    # --------------------------RANSAC Outlier Removal----------------------------------
    # F_hat, mask_F = cv2.findFundamentalMat(kp_a, kp_b, method=cv2.USAC_ACCURATE,
    #                                        ransacReprojThreshold=1, confidence=0.99)
    F_hat, mask_F = cv2.findFundamentalMat(kp_a, kp_b, method=cv2.USAC_MAGSAC,
                                           ransacReprojThreshold=1, confidence=0.99)
    toc = time.time()
    tt1 = toc - tic

    if mask_F is not None:
        mask_F = mask_F[:, 0].astype(bool)
    else:
        mask_F = np.zeros_like(kp_a[:, 0]).astype(bool)

    points0 = kp_a[mask_F]
    points1 = kp_b[mask_F]
    # canvas = demo_utils.draw_match(img0, img1, points0, points1)
    canvas = demo_utils.draw_match(img0, img1, kp_a[mask_F], kp_a[mask_F])

    # -----------------------------------------------------------------------
    putative_num = len(kpts0)
    correct_num = len(kpts0[mask_F])
    inliner_ratio = correct_num / putative_num
    text1 = "putative_num:{}".format(putative_num)
    text2 = 'correct_num:{}'.format(correct_num)
    text3 = 'inliner ratio:%.3f' % inliner_ratio
    text4 = 'run time: %.3fs' % tt1

    print('putative_num:{}'.format(putative_num), '\ncorrect_num:{}'.format(correct_num),
          '\ninliner ratio:%.3f' % inliner_ratio, '\nrun time: %.3fs' % tt1)

    cv2.putText(canvas, str(text1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(canvas, str(text2), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(canvas, str(text3), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(canvas, str(text4), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, canvas)

    flops, params = profile(engine, inputs=(img0, img1))
    print("Params：", "%.2f" % (params / (1000 ** 2)), "M")
    print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))

    # Params： 29.35M
    # GFLOPS： 452.23

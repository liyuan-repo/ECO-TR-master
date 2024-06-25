import matplotlib.pyplot as plt
from dotmap import DotMap
from src.config.default import get_cfg_defaults
from src.models.Reconstruction_func import *
from src.models.ecotr_engines import *
from src.models.utils.utils import *
from src.utils.misc import lower_config

if __name__ == '__main__':
    # -------------------------Load inference engine-----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dict = DotMap(lower_config(get_cfg_defaults()))
    dict.ecotr.engine.device = device
    fix_randomness(42)
    engine = ECOTR_Engine(dict.ecotr)

    # -------------------------Select image pairs--------------------------
    torch.cuda.empty_cache()

    pairs = [['../assets/others/34256848_2093205455.jpg', '../assets/others/63163121_2253976476.jpg'],  # taj
             ['../assets/others/recon_img_0.jpg', '../assets/others/recon_img_1.jpg']]

    camera = [['../assets/others/calibration_000157.npy', '../assets/others/calibration_000128.npy'],
              ['../assets/others/camera_0.npy', '../assets/others/camera_1.npy']]

    pair_idx = 1
    # img0 = cv2.imread(pairs[pair_idx][0])[..., [2, 1, 0]]
    # img1 = cv2.imread(pairs[pair_idx][1])[..., [2, 1, 0]]
    # img0 = cv2.imread('../assets/NIR01.jpg')
    # img1 = cv2.imread('../assets/photo.jpg')
    # img0 = cv2.imread("../assets/4SARSets/pair2-1.tif")
    # img1 = cv2.imread("../assets/4SARSets/pair2-2.tif")
    # img0 = cv2.imread('../assets//others/recon_img_0.jpg')
    # img1 = cv2.imread('../assets/others/recon_img_1.jpg')
    img0 = cv2.imread("../assets/3MapSets/pair1-2.jpg")
    img1 = cv2.imread("../assets/3MapSets/pair1-1.jpg")
    # camera_a = np.load(camera[pair_idx][0], allow_pickle=True).item()
    # camera_b = np.load(camera[pair_idx][1], allow_pickle=True).item()

    # -----------------------Func1. Coarse to fine Inference------------------

    engine.MAX_KPTS_NUM = 4000
    engine.ASPECT_RATIOS = [1.0, 1.5]

    matches = engine.forward(img0, img1, cycle=True, level='fine')
    # matches=engine.forward_2stage(img0,img1,cycle=False)
    # refined_matches=engine.forward_refine(img0,img1,queries = matches[:,:2].reshape(-1,2),
    # corrs = matches[:,2:4].reshape(-1,2))

    mask = matches[:, -1] < 1e-2
    matches = matches[mask]
    canvas = draw_matches(img0, img1, matches.copy(), 'dot')
    height, width = canvas.shape[:2]

    dpi = 100
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    plt.show()

    # center_a = camera_a['cam_center']
    # center_b = camera_b['cam_center']
    #
    # rays_a = PointCloudProjector.pcd_2d_to_pcd_3d_np(matches[:, :2], np.ones([matches.shape[0], 1]) * 2,
    #                                                  camera_a['intrinsic'], motion=camera_a['c2w'])
    # rays_b = PointCloudProjector.pcd_2d_to_pcd_3d_np(matches[:, 2:4], np.ones([matches.shape[0], 1]) * 2,
    #                                                  camera_b['intrinsic'], motion=camera_b['c2w'])
    #
    # dir_a = rays_a - center_a
    # dir_b = rays_b - center_b
    # center_a = np.array([center_a] * matches.shape[0])
    # center_b = np.array([center_b] * matches.shape[0])
    # points = triangulate_rays_to_pcd(center_a, dir_a, center_b, dir_b)
    # colors = ((img0[tuple(np.floor(matches[:, :2]).astype(int)[:, ::-1].T)] / 255 + img1[
    #     tuple(np.floor(matches[:, 2:4]).astype(int)[:, ::-1].T)] / 255) / 2) * 255
    # colors = np.array(colors).astype(np.uint8)[..., ::-1]
    #
    # toPLY(points, colors, './recon.ply')

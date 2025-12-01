import cv2
import numpy as np
import os
from skimage.filters.rank import entropy
from skimage.morphology import disk


def compute_saliency(img):
    """
    计算Laplacian熵显著性图
    
    Args:
        img: 输入图像 (H, W) - 灰度图
    
    Returns:
        显著性图 (H, W)
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 计算Laplacian算子（二阶导数，检测边缘）
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # 计算熵（使用局部窗口）
    entropy_map = entropy(np.abs(laplacian).astype(np.uint8), disk(3))
    
    return entropy_map


def butterworth_highpass(img, d0=30, n=2):
    """
    对单帧图像进行频域Butterworth高通滤波
    
    Args:
        img: 单帧图像 (H, W) - 灰度图
        d0: 截止频率（默认30，增大可保留更多低频信息，减小可过滤更多低频）
        n: 滤波器阶数（默认2，增大可使过渡更陡峭）
    
    Returns:
        滤波后的图像 (H, W)
    """
    h, w = img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    du, dv = u - w//2, v - h//2
    D = np.sqrt(du**2 + dv**2)
    
    # 计算Butterworth高通滤波器
    H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))
    
    # 频域变换
    img_dft = np.fft.fftshift(np.fft.fft2(img))
    img_hp = np.real(np.fft.ifft2(np.fft.ifftshift(img_dft * H)))
    
    return img_hp


def motion_compensate(frame1, frame2, 
                      motion_distance_threshold=50,
                      min_tracking_points=15,
                      ransac_threshold=1.0,
                      klt_epsilon=0.003,
                      klt_max_iter=30):
    """
    基于网格的KLT光流跟踪进行运动补偿
    
    Args:
        frame1: 前一帧图像（灰度图）
        frame2: 当前帧图像（灰度图）
        motion_distance_threshold: 运动距离阈值，超过此值的点会被过滤（默认50，增大可保留更多点，减小可过滤噪声）
        min_tracking_points: 最小跟踪点数量，少于此时使用单位矩阵（默认15，减小可降低要求）
        ransac_threshold: RANSAC重投影误差阈值（默认3.0，减小可提高精度但对噪声更敏感）
        klt_epsilon: KLT光流精度阈值（默认0.003，减小可提高对微弱运动的敏感度）
        klt_max_iter: KLT光流最大迭代次数（默认30）
    
    Returns:
        compensated: 运动补偿后的图像
        mask: 掩膜
        avg_dst: 平均运动距离
        motion_x: 平均x方向运动
        motion_y: 平均y方向运动
        homography_matrix: 单应性矩阵
    """
    # KLT光流跟踪参数
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, klt_max_iter, klt_epsilon))

    width = frame2.shape[1]
    height = frame2.shape[0]
    scale = 4  # 放大倍数，提高跟踪精度

    # 将图像放大以提高跟踪精度（自适应输入图像尺寸）
    new_width = int(width * scale)
    new_height = int(height * scale)
    frame1_grid = cv2.resize(frame1, (new_width, new_height), dst=None, interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (new_width, new_height), dst=None, interpolation=cv2.INTER_CUBIC)

    width_grid = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    
    # 生成网格点
    gridSizeW = 32 * 4
    gridSizeH = 24 * 4
    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), 
                     np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    # KLT光流跟踪
    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    # 选择有效的跟踪点
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    # 计算运动距离和方向，过滤异常点
    motion_distance = []
    translate_x = []
    translate_y = []
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))

        # 过滤运动距离过大的点（可能是噪声）
        # 注意：对于微弱运动目标，可以增大此阈值或设为None来保留所有点
        if motion_distance_threshold is not None and motion_distance0 > motion_distance_threshold:
            continue

        translate_x0 = a - c
        translate_y0 = b - d

        motion_distance.append(motion_distance0)
        translate_x.append(translate_x0)
        translate_y.append(translate_y0)
    
    motion_dist = np.array(motion_distance)
    motion_x = np.mean(np.array(translate_x)) if len(translate_x) > 0 else 0
    motion_y = np.mean(np.array(translate_y)) if len(translate_y) > 0 else 0
    avg_dst = np.mean(motion_dist) if len(motion_dist) > 0 else 0

    # 计算单应性矩阵
    if len(good_old) < min_tracking_points:
        # 如果跟踪点太少，使用单位矩阵（几乎不变换）
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        # 使用RANSAC算法计算单应性矩阵
        # ransac_threshold: 重投影误差阈值，减小可提高精度但对噪声更敏感
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, ransac_threshold)

    # 根据变换矩阵计算变换之后的图像（运动补偿）
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), 
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜（用于标记变换后的有效区域）
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix


def energy_accumulation(saliency_maps, alpha=0.8):
    """
    多帧能量累积，增强目标信号
    
    Args:
        saliency_maps: 显著性图序列 [frames, (H, W)] 或列表
        alpha: 衰减系数（默认0.8，范围0-1，越大越重视新帧）
    
    Returns:
        累积能量图 (H, W)
    """
    accumulated_energy = np.zeros_like(saliency_maps[0])
    
    for t, smap in enumerate(saliency_maps):
        accumulated_energy += (alpha**t) * (smap.astype(np.float32)**2)
    
    # 归一化
    max_val = np.max(accumulated_energy)
    if max_val > 0:
        accumulated_energy = accumulated_energy / (max_val + 1e-8)
    
    return accumulated_energy


def background_suppression(energy_map, threshold=0.1):
    """
    背景抑制，减少噪声干扰
    
    Args:
        energy_map: 能量图 (H, W)，值范围[0, 1]
        threshold: 阈值（默认0.1，小于此值的区域被抑制）
    
    Returns:
        抑制后的能量图 (H, W)，uint8格式，值范围[0, 255]
    """
    # 阈值处理
    energy_map = energy_map.copy()
    energy_map[energy_map < threshold] = 0
    
    # 中值滤波去噪
    energy_map_uint8 = (energy_map * 255).astype(np.uint8)
    suppressed = cv2.medianBlur(energy_map_uint8, 3)
    
    return suppressed


def generate_motion_diff_map(img1_path, img2_path, img3_path=None, output_path=None, show_result=True,
                              motion_distance_threshold=50,
                              min_tracking_points=15,
                              ransac_threshold=3.0,
                              klt_epsilon=0.003,
                              klt_max_iter=30,
                              use_highpass=False,
                              highpass_d0=30,
                              highpass_n=2,
                              use_saliency=False,
                              saliency_weight=0.5,
                              energy_alpha=0.8,
                              suppression_threshold=0.1):
    """
    生成Motion Difference Map
    
    Args:
        img1_path: 第一张图片路径（前一帧）
        img2_path: 第二张图片路径（当前帧）
        output_path: 输出路径（可选，如果不指定则自动生成）
        show_result: 是否显示结果
        motion_distance_threshold: 运动距离阈值，超过此值的点会被过滤（默认50）
                                   - 对于微弱运动：可增大到100-200或设为None保留所有点
                                   - 对于强运动：可减小到20-30过滤噪声
        min_tracking_points: 最小跟踪点数量（默认15，减小可降低要求）
        ransac_threshold: RANSAC重投影误差阈值（默认3.0）
                         - 对于微弱运动：可减小到1.0-2.0提高精度
        klt_epsilon: KLT光流精度阈值（默认0.003）
                    - 对于微弱运动：可减小到0.001-0.002提高敏感度
        klt_max_iter: KLT光流最大迭代次数（默认30）
        use_highpass: 是否使用高频滤波（默认False）
                     - True: 启用Butterworth高通滤波，突出高频信息（边缘、细节）
                     - False: 不使用高频滤波
        highpass_d0: 高通滤波截止频率（默认30）
                    - 增大：保留更多低频信息
                    - 减小：过滤更多低频，只保留高频（边缘、细节）
        highpass_n: 高通滤波器阶数（默认2）
                   - 增大：过渡更陡峭，滤波效果更明显
        use_saliency: 是否使用显著性计算（默认False）
                     - True: 计算显著性图并与Motion Difference Map融合
                     - False: 不使用显著性
        saliency_weight: 显著性权重（默认0.5）
                        - 范围[0, 1]，控制显著性图在融合中的比重
                        - 0: 完全不使用显著性
                        - 1: 完全使用显著性
                        - 0.5: 等权重融合
        energy_alpha: 能量累积衰减系数（默认0.8）
                     - 范围[0, 1]，越大越重视新帧
        suppression_threshold: 背景抑制阈值（默认0.1）
                              - 小于此值的区域被抑制
    
    Returns:
        motion_diff_map: Motion Difference Map（numpy数组）
        saliency_map: 显著性图（如果use_saliency=True，否则为None）
    """
    # 读取图像
    print(f"读取图像1 (t-2): {img1_path}")
    frame1 = cv2.imread(img1_path)
    if frame1 is None:
        raise ValueError(f"无法读取图像: {img1_path}")
    
    print(f"读取图像2 (t-1): {img2_path}")
    frame2 = cv2.imread(img2_path)
    if frame2 is None:
        raise ValueError(f"无法读取图像: {img2_path}")
    
    # 读取第三帧（如果提供）
    if img3_path is not None:
        print(f"读取图像3 (t): {img3_path}")
        frame3 = cv2.imread(img3_path)
        if frame3 is None:
            raise ValueError(f"无法读取图像: {img3_path}")
        use_three_frames = True
    else:
        frame3 = None
        use_three_frames = False
        print("未提供第三帧图像，使用两帧模式")
    
    print(f"图像1尺寸: {frame1.shape}")
    print(f"图像2尺寸: {frame2.shape}")
    if use_three_frames:
        print(f"图像3尺寸: {frame3.shape}")
    
    # 确保所有图像尺寸相同
    target_shape = frame1.shape[:2]  # (H, W)
    if frame2.shape[:2] != target_shape:
        print("警告: 图像2尺寸不同，将调整为图像1的尺寸")
        frame2 = cv2.resize(frame2, (target_shape[1], target_shape[0]))
    if use_three_frames and frame3.shape[:2] != target_shape:
        print("警告: 图像3尺寸不同，将调整为图像1的尺寸")
        frame3 = cv2.resize(frame3, (target_shape[1], target_shape[0]))
    
    # 预处理：高斯模糊去噪
    # print("进行高斯模糊去噪...")
    # frame1_blur = cv2.GaussianBlur(frame1, (11, 11), 0)
    # frame2_blur = cv2.GaussianBlur(frame2, (11, 11), 0)
    
    # 转换为灰度图
    print("转换为灰度图...")
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    if use_three_frames:
        frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    
    # 高频滤波（可选）- 对所有帧进行滤波
    if use_highpass:
        print(f"进行Butterworth高通滤波 (d0={highpass_d0}, n={highpass_n})...")
        # 保存原始图像用于对比
        frame1_original = frame1_gray.copy()
        frame2_original = frame2_gray.copy()
        
        # 应用高频滤波
        frame1_gray = butterworth_highpass(frame1_gray, d0=highpass_d0, n=highpass_n)
        frame2_gray = butterworth_highpass(frame2_gray, d0=highpass_d0, n=highpass_n)
        if use_three_frames:
            frame3_gray = butterworth_highpass(frame3_gray, d0=highpass_d0, n=highpass_n)
        
        # 归一化到0-255范围（频域滤波可能产生负值或超出范围的值）
        def normalize_frame(frame):
            frame_min, frame_max = frame.min(), frame.max()
            if frame_max > frame_min:
                return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                return np.clip(frame, 0, 255).astype(np.uint8)
        
        frame1_gray = normalize_frame(frame1_gray)
        frame2_gray = normalize_frame(frame2_gray)
        if use_three_frames:
            frame3_gray = normalize_frame(frame3_gray)
        
        print("高频滤波完成")
    else:
        print("跳过高频滤波")
    
    # 计算Motion Difference Maps
    motion_diff_maps = []
    
    if use_three_frames:
        # 三帧模式：计算两个Motion Difference Map
        print("="*60)
        print("三帧模式：计算两个Motion Difference Map")
        print("="*60)
        
        # 1. 计算帧t-2和t-1的Motion Difference Map
        print("\n计算帧t-2和t-1的Motion Difference Map...")
        print(f"参数设置: 运动距离阈值={motion_distance_threshold}, "
              f"最小跟踪点数={min_tracking_points}, "
              f"RANSAC阈值={ransac_threshold}, "
              f"KLT精度={klt_epsilon}")
        img_compensate_12, mask12, avg_dist12, motion_x12, motion_y12, homo_matrix12 = motion_compensate(
            frame1_gray, frame2_gray,
            motion_distance_threshold=motion_distance_threshold,
            min_tracking_points=min_tracking_points,
            ransac_threshold=ransac_threshold,
            klt_epsilon=klt_epsilon,
            klt_max_iter=klt_max_iter)
        
        frameDiff_12 = cv2.absdiff(frame2_gray, img_compensate_12)
        print(f"帧t-2和t-1: 平均运动距离={avg_dist12:.2f}, x={motion_x12:.2f}, y={motion_y12:.2f}")
        
        # 2. 计算帧t-1和t的Motion Difference Map
        print("\n计算帧t-1和t的Motion Difference Map...")
        img_compensate_23, mask23, avg_dist23, motion_x23, motion_y23, homo_matrix23 = motion_compensate(
            frame2_gray, frame3_gray,
            motion_distance_threshold=motion_distance_threshold,
            min_tracking_points=min_tracking_points,
            ransac_threshold=ransac_threshold,
            klt_epsilon=klt_epsilon,
            klt_max_iter=klt_max_iter)
        
        frameDiff_23 = cv2.absdiff(frame3_gray, img_compensate_23)
        print(f"帧t-1和t: 平均运动距离={avg_dist23:.2f}, x={motion_x23:.2f}, y={motion_y23:.2f}")
        
        # 保存用于显示的初始Motion Difference Maps
        frameDiff_initial_12 = frameDiff_12.copy()
        frameDiff_initial_23 = frameDiff_23.copy()
        
        # 3. 对每个Motion Difference Map计算显著性（如果启用）
        saliency_maps = []
        if use_saliency:
            print("\n对Motion Difference Map计算显著性...")
            # 对第一个Motion Difference Map计算显著性
            print("计算帧t-2和t-1的显著性图...")
            saliency_12 = compute_saliency(frameDiff_12)
            # 归一化显著性图到0-255范围
            saliency_12_min, saliency_12_max = saliency_12.min(), saliency_12.max()
            if saliency_12_max > saliency_12_min:
                saliency_12 = ((saliency_12 - saliency_12_min) / (saliency_12_max - saliency_12_min) * 255).astype(np.uint8)
            else:
                saliency_12 = np.clip(saliency_12, 0, 255).astype(np.uint8)
            saliency_maps.append(saliency_12.astype(np.float32) / 255.0)  # 归一化到[0,1]
            print(f"帧t-2和t-1显著性图范围: [{saliency_12_min:.2f}, {saliency_12_max:.2f}]")
            
            # 对第二个Motion Difference Map计算显著性
            print("计算帧t-1和t的显著性图...")
            saliency_23 = compute_saliency(frameDiff_23)
            # 归一化显著性图到0-255范围
            saliency_23_min, saliency_23_max = saliency_23.min(), saliency_23.max()
            if saliency_23_max > saliency_23_min:
                saliency_23 = ((saliency_23 - saliency_23_min) / (saliency_23_max - saliency_23_min) * 255).astype(np.uint8)
            else:
                saliency_23 = np.clip(saliency_23, 0, 255).astype(np.uint8)
            saliency_maps.append(saliency_23.astype(np.float32) / 255.0)  # 归一化到[0,1]
            print(f"帧t-1和t显著性图范围: [{saliency_23_min:.2f}, {saliency_23_max:.2f}]")
            
            # 保存用于显示的显著性图
            saliency_12_display = saliency_12.copy()
            saliency_23_display = saliency_23.copy()
            saliency_map = None  # 三帧模式下没有单一显著性图
        else:
            # 如果不使用显著性，直接使用Motion Difference Map
            print("\n跳过显著性计算，直接使用Motion Difference Map进行融合")
            motion_diff_maps.append(frameDiff_12.astype(np.float32) / 255.0)  # 归一化到[0,1]
            motion_diff_maps.append(frameDiff_23.astype(np.float32) / 255.0)  # 归一化到[0,1]
            saliency_12_display = None
            saliency_23_display = None
            saliency_map = None
        
        # 4. 能量累积融合
        print(f"\n进行能量累积融合 (alpha={energy_alpha})...")
        if use_saliency:
            # 对显著性图进行能量累积
            accumulated_energy = energy_accumulation(saliency_maps, alpha=energy_alpha)
        else:
            # 对Motion Difference Map进行能量累积
            accumulated_energy = energy_accumulation(motion_diff_maps, alpha=energy_alpha)
        print(f"能量累积完成，能量图范围: [{accumulated_energy.min():.4f}, {accumulated_energy.max():.4f}]")
        
        # 5. 背景抑制
        print(f"进行背景抑制 (threshold={suppression_threshold})...")
        frameDiff = background_suppression(accumulated_energy, threshold=suppression_threshold)
        print("背景抑制完成")
        
        # 保存用于显示的累积能量图（抑制前）
        accumulated_energy_display = (accumulated_energy * 255).astype(np.uint8)
        
        # 确保saliency变量在可视化部分可访问（初始化默认值）
        if not use_saliency:
            saliency_12_display = None
            saliency_23_display = None
        
    else:
        # 两帧模式（保持原有逻辑）
        print("="*60)
        print("两帧模式")
        print("="*60)
        print("进行运动补偿...")
        print(f"参数设置: 运动距离阈值={motion_distance_threshold}, "
              f"最小跟踪点数={min_tracking_points}, "
              f"RANSAC阈值={ransac_threshold}, "
              f"KLT精度={klt_epsilon}")
        img_compensate, mask, avg_dist, motion_x, motion_y, homo_matrix = motion_compensate(
            frame1_gray, frame2_gray,
            motion_distance_threshold=motion_distance_threshold,
            min_tracking_points=min_tracking_points,
            ransac_threshold=ransac_threshold,
            klt_epsilon=klt_epsilon,
            klt_max_iter=klt_max_iter)
        
        print(f"平均运动距离: {avg_dist:.2f}")
        print(f"平均x方向运动: {motion_x:.2f}")
        print(f"平均y方向运动: {motion_y:.2f}")
        
        # 计算Motion Difference Map（初始版本，融合显著性之前）
        print("计算Motion Difference Map...")
        frameDiff_initial = cv2.absdiff(frame2_gray, img_compensate)  # 保存初始版本
        frameDiff = frameDiff_initial.copy()  # 用于后续融合
        frameDiff_initial_12 = frameDiff_initial.copy()
        frameDiff_initial_23 = None
        accumulated_energy_display = None
    
    # 显著性计算（可选）- 仅在两帧模式下使用
    saliency_map = None
    if use_saliency and not use_three_frames:
        print("计算显著性图（基于初始Motion Difference Map）...")
        # 对初始Motion Difference Map计算显著性
        saliency_map = compute_saliency(frameDiff_initial)
        
        # 归一化显著性图到0-255范围
        saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
        if saliency_max > saliency_min:
            saliency_map = ((saliency_map - saliency_min) / (saliency_max - saliency_min) * 255).astype(np.uint8)
        else:
            saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)
        
        # 融合Motion Difference Map和显著性图
        print(f"融合Motion Difference Map和显著性图 (权重={saliency_weight})...")
        frameDiff_float = frameDiff.astype(np.float32)
        saliency_float = saliency_map.astype(np.float32)
        
        # 加权融合
        frameDiff = ((1 - saliency_weight) * frameDiff_float + 
                     saliency_weight * saliency_float).astype(np.uint8)
        
        print(f"显著性图范围: [{saliency_min:.2f}, {saliency_max:.2f}]")
        print("显著性融合完成")
    else:
        print("跳过显著性计算")
    
    # 保存结果
    if output_path is None:
        # 自动生成输出路径
        base_dir = os.path.dirname(img2_path)
        base_name = os.path.splitext(os.path.basename(img2_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_motion_diff.jpg")
    
    # 保存最终融合后的Motion Difference Map
    print(f"保存最终结果到: {output_path}")
    cv2.imwrite(output_path, frameDiff)
    
    # 保存中间结果图像
    base_dir = os.path.dirname(img2_path)
    base_name = os.path.splitext(os.path.basename(img2_path))[0]
    
    if use_three_frames:
        # 三帧模式：保存所有中间结果
        filter_label = "filtered" if use_highpass else "gray"
        # 1. 保存高频滤波后的图像（如果使用了高频滤波）
        frame1_path = os.path.join(base_dir, f"{base_name}_frame1_{filter_label}.jpg")
        frame2_path = os.path.join(base_dir, f"{base_name}_frame2_{filter_label}.jpg")
        frame3_path = os.path.join(base_dir, f"{base_name}_frame3_{filter_label}.jpg")
        cv2.imwrite(frame1_path, frame1_gray)
        cv2.imwrite(frame2_path, frame2_gray)
        cv2.imwrite(frame3_path, frame3_gray)
        print(f"保存Frame1: {frame1_path}")
        print(f"保存Frame2: {frame2_path}")
        print(f"保存Frame3: {frame3_path}")
        
        # 2. 保存两个Motion Difference Map
        diff_12_path = os.path.join(base_dir, f"{base_name}_motion_diff_12.jpg")
        diff_23_path = os.path.join(base_dir, f"{base_name}_motion_diff_23.jpg")
        cv2.imwrite(diff_12_path, frameDiff_initial_12)
        cv2.imwrite(diff_23_path, frameDiff_initial_23)
        print(f"保存Motion Diff (t-2,t-1): {diff_12_path}")
        print(f"保存Motion Diff (t-1,t): {diff_23_path}")
        
        # 3. 保存显著性图（如果计算了）
        if use_saliency and saliency_12_display is not None:
            saliency_12_path = os.path.join(base_dir, f"{base_name}_saliency_12.jpg")
            saliency_23_path = os.path.join(base_dir, f"{base_name}_saliency_23.jpg")
            cv2.imwrite(saliency_12_path, saliency_12_display)
            cv2.imwrite(saliency_23_path, saliency_23_display)
            print(f"保存显著性图 (t-2,t-1): {saliency_12_path}")
            print(f"保存显著性图 (t-1,t): {saliency_23_path}")
        
        # 4. 保存能量累积图（背景抑制前）
        energy_path = os.path.join(base_dir, f"{base_name}_energy_accumulated.jpg")
        cv2.imwrite(energy_path, accumulated_energy_display)
        print(f"保存能量累积图: {energy_path}")
    else:
        # 两帧模式：保存中间结果
        # 1. 保存高频滤波后的图像（如果使用了高频滤波）
        if use_highpass:
            frame1_filtered_path = os.path.join(base_dir, f"{base_name}_frame1_filtered.jpg")
            frame2_filtered_path = os.path.join(base_dir, f"{base_name}_frame2_filtered.jpg")
            cv2.imwrite(frame1_filtered_path, frame1_gray)
            cv2.imwrite(frame2_filtered_path, frame2_gray)
            print(f"保存滤波后的Frame1: {frame1_filtered_path}")
            print(f"保存滤波后的Frame2: {frame2_filtered_path}")
        else:
            # 如果没有使用高频滤波，保存原始灰度图
            frame1_filtered_path = os.path.join(base_dir, f"{base_name}_frame1_gray.jpg")
            frame2_filtered_path = os.path.join(base_dir, f"{base_name}_frame2_gray.jpg")
            cv2.imwrite(frame1_filtered_path, frame1_gray)
            cv2.imwrite(frame2_filtered_path, frame2_gray)
            print(f"保存灰度图Frame1: {frame1_filtered_path}")
            print(f"保存灰度图Frame2: {frame2_filtered_path}")
        
        # 2. 保存初始Motion Difference Map（融合显著性之前）
        frameDiff_initial_path = os.path.join(base_dir, f"{base_name}_motion_diff_initial.jpg")
        cv2.imwrite(frameDiff_initial_path, frameDiff_initial_12)
        print(f"保存初始Motion Difference Map: {frameDiff_initial_path}")
        
        # 3. 保存显著性图（如果计算了）
        if use_saliency and saliency_map is not None:
            saliency_path = os.path.join(base_dir, f"{base_name}_saliency.jpg")
            cv2.imwrite(saliency_path, saliency_map)
            print(f"保存显著性图: {saliency_path}")
    
    print("所有图像保存完成!")
    
    # 显示结果
    if show_result:
        # 创建对比显示 - 展示所有中间处理步骤
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (0, 255, 0)
        thickness = 2
        
        # 调整大小以便显示（如果图像太大）
        max_display_size = 500
        scale = min(1.0, max_display_size / max(frame1.shape[0], frame1.shape[1]))
        h, w = int(frame1.shape[0] * scale), int(frame1.shape[1] * scale)
        
        if use_three_frames:
            # 三帧模式可视化
            # 将灰度图转换为BGR以便显示
            frame1_display = cv2.resize(cv2.cvtColor(frame1_gray, cv2.COLOR_GRAY2BGR), (w, h))
            frame2_display = cv2.resize(cv2.cvtColor(frame2_gray, cv2.COLOR_GRAY2BGR), (w, h))
            frame3_display = cv2.resize(cv2.cvtColor(frame3_gray, cv2.COLOR_GRAY2BGR), (w, h))
            diff_12_display = cv2.resize(cv2.cvtColor(frameDiff_initial_12, cv2.COLOR_GRAY2BGR), (w, h))
            diff_23_display = cv2.resize(cv2.cvtColor(frameDiff_initial_23, cv2.COLOR_GRAY2BGR), (w, h))
            energy_display = cv2.resize(cv2.cvtColor(accumulated_energy_display, cv2.COLOR_GRAY2BGR), (w, h))
            diff_final_display = cv2.resize(cv2.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR), (w, h))
            
            # 检查是否使用了显著性并显示
            # saliency_12_display和saliency_23_display在三帧模式的计算部分已经定义
            if use_saliency and saliency_12_display is not None:
                # 如果使用了显著性，显示显著性图
                saliency_12_display_img = cv2.resize(cv2.cvtColor(saliency_12_display, cv2.COLOR_GRAY2BGR), (w, h))
                saliency_23_display_img = cv2.resize(cv2.cvtColor(saliency_23_display, cv2.COLOR_GRAY2BGR), (w, h))
                
                # 创建3行x3列布局
                # 第一行：Frame1 (t-2), Frame2 (t-1), Frame3 (t) - 滤波后的三张图像
                # 第二行：Motion Diff (t-2,t-1), Motion Diff (t-1,t), 空白 - 两张motion_difference_map
                # 第三行：Saliency (t-2,t-1), Saliency (t-1,t), Final Result - 两张saliency和final result
                top_row = np.hstack([frame1_display, frame2_display, frame3_display])
                second_row = np.hstack([diff_12_display, diff_23_display, np.zeros_like(frame1_display)])
                third_row = np.hstack([saliency_12_display_img, saliency_23_display_img, diff_final_display])
                combined = np.vstack([top_row, second_row, third_row])
                
                # 添加标签
                h_img = combined.shape[0] // 3
                w_img = combined.shape[1] // 3
                
                filter_label = " (Filtered)" if use_highpass else ""
                # 第一行标签
                cv2.putText(combined, f"Frame t-2{filter_label}", (10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame t-1{filter_label}", (w_img + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame t{filter_label}", (w_img * 2 + 10, 25), font, font_scale, color, thickness)
                
                # 第二行标签
                cv2.putText(combined, "Motion Diff (t-2,t-1)", (10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (t-1,t)", (w_img + 10, h_img + 25), font, font_scale, color, thickness)
                
                # 第三行标签
                cv2.putText(combined, "Saliency (t-2,t-1)", (10, h_img * 2 + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Saliency (t-1,t)", (w_img + 10, h_img * 2 + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Final Result", (w_img * 2 + 10, h_img * 2 + 25), font, font_scale, color, thickness)
                
                window_title = "Three-Frame Motion Difference Map [Saliency Enhanced]"
                
                cv2.imshow(window_title, combined)
                print("\n" + "="*60)
                print("可视化窗口说明（三帧模式）:")
                print("="*60)
                print("第一行: Frame t-2 (滤波后), Frame t-1 (滤波后), Frame t (滤波后)")
                print("第二行: Motion Diff (t-2,t-1), Motion Diff (t-1,t), 空白")
                print("第三行: Saliency (t-2,t-1), Saliency (t-1,t), Final Result (背景抑制后)")
                print("="*60)
            else:
                # 不使用显著性时的布局（保持原有3行布局）
                # 第一行：Frame1 (t-2), Frame2 (t-1), Frame3 (t)
                # 第二行：Motion Diff (t-2,t-1), Motion Diff (t-1,t), Energy Accumulated
                # 第三行：空白, 空白, Final Result
                top_row = np.hstack([frame1_display, frame2_display, frame3_display])
                middle_row = np.hstack([diff_12_display, diff_23_display, energy_display])
                blank = np.zeros_like(frame1_display)
                bottom_row = np.hstack([blank, blank, diff_final_display])
                combined = np.vstack([top_row, middle_row, bottom_row])
                
                # 添加标签
                h_img = combined.shape[0] // 3
                w_img = combined.shape[1] // 3
                
                filter_label = " (Filtered)" if use_highpass else ""
                # 第一行标签
                cv2.putText(combined, f"Frame t-2{filter_label}", (10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame t-1{filter_label}", (w_img + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame t{filter_label}", (w_img * 2 + 10, 25), font, font_scale, color, thickness)
                
                # 第二行标签
                cv2.putText(combined, "Motion Diff (t-2,t-1)", (10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (t-1,t)", (w_img + 10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Energy Accumulated", (w_img * 2 + 10, h_img + 25), font, font_scale, color, thickness)
                
                # 第三行标签
                cv2.putText(combined, "Final Result", (w_img * 2 + 10, h_img * 2 + 25), font, font_scale, color, thickness)
                
                window_title = "Three-Frame Motion Difference Map"
                if use_highpass:
                    window_title += " [Highpass Filtered]"
                
                cv2.imshow(window_title, combined)
                print("\n" + "="*60)
                print("可视化窗口说明（三帧模式）:")
                print("="*60)
                print("第一行: Frame t-2 (滤波后), Frame t-1 (滤波后), Frame t (滤波后)")
                print("第二行: Motion Diff (t-2,t-1), Motion Diff (t-1,t), Energy Accumulated (能量累积)")
                print("第三行: 空白, 空白, Final Result (背景抑制后)")
                print("="*60)
            
        else:
            # 两帧模式可视化（保持原有逻辑）
            frame1_display = cv2.resize(cv2.cvtColor(frame1_gray, cv2.COLOR_GRAY2BGR), (w, h))
            frame2_display = cv2.resize(cv2.cvtColor(frame2_gray, cv2.COLOR_GRAY2BGR), (w, h))
            compensated_display = cv2.resize(cv2.cvtColor(img_compensate, cv2.COLOR_GRAY2BGR), (w, h))
            diff_initial_display = cv2.resize(cv2.cvtColor(frameDiff_initial_12, cv2.COLOR_GRAY2BGR), (w, h))
            diff_final_display = cv2.resize(cv2.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR), (w, h))
            
            # 如果使用了显著性，也显示显著性图
            if use_saliency and saliency_map is not None:
                saliency_display = cv2.resize(cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR), (w, h))
            
            # 创建组合图像
            if use_saliency and saliency_map is not None:
                # 6宫格布局
                top_row = np.hstack([frame1_display, frame2_display, compensated_display])
                bottom_row = np.hstack([saliency_display, diff_initial_display, diff_final_display])
                combined = np.vstack([top_row, bottom_row])
                
                h_img = combined.shape[0] // 2
                w_img = combined.shape[1] // 3
                
                filter_label = " (Filtered)" if use_highpass else ""
                cv2.putText(combined, f"Frame1{filter_label}", (10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame2{filter_label}", (w_img + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Compensated", (w_img * 2 + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Saliency Map", (10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (Initial)", (w_img + 10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (Final)", (w_img * 2 + 10, h_img + 25), font, font_scale, color, thickness)
            else:
                # 5宫格布局
                top_row = np.hstack([frame1_display, frame2_display, compensated_display])
                blank = np.zeros_like(frame1_display)
                bottom_row = np.hstack([blank, diff_initial_display, diff_final_display])
                combined = np.vstack([top_row, bottom_row])
                
                h_img = combined.shape[0] // 2
                w_img = combined.shape[1] // 3
                
                filter_label = " (Filtered)" if use_highpass else ""
                cv2.putText(combined, f"Frame1{filter_label}", (10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, f"Frame2{filter_label}", (w_img + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Compensated", (w_img * 2 + 10, 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (Initial)", (w_img + 10, h_img + 25), font, font_scale, color, thickness)
                cv2.putText(combined, "Motion Diff (Final)", (w_img * 2 + 10, h_img + 25), font, font_scale, color, thickness)
            
            window_title = "Two-Frame Motion Difference Map"
            if use_highpass:
                window_title += " [Highpass Filtered]"
            if use_saliency:
                window_title += " [Saliency Enhanced]"
            
            cv2.imshow(window_title, combined)
            print("\n" + "="*60)
            print("可视化窗口说明（两帧模式）:")
            print("="*60)
            print("第一行: Frame1 (滤波后), Frame2 (滤波后), Compensated (运动补偿后)")
            print("第二行:")
            if use_saliency and saliency_map is not None:
                print("  - Saliency Map: 显著性图")
                print("  - Motion Diff (Initial): 初始Motion Difference Map")
                print("  - Motion Diff (Final): 最终融合结果")
            else:
                print("  - Motion Diff (Initial): 初始Motion Difference Map")
                print("  - Motion Diff (Final): 最终结果")
            print("="*60)
        
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 返回值处理
    if use_three_frames:
        if use_saliency:
            # 三帧模式 + 显著性：返回最终结果和两个显著性图
            return frameDiff, saliency_12_display, saliency_23_display
        else:
            # 三帧模式：只返回最终结果
            return frameDiff
    else:
        # 两帧模式
        if use_saliency:
            return frameDiff, saliency_map
        else:
            return frameDiff


if __name__ == "__main__":
    # 测试图片路径
    # 三帧模式：需要提供三张连续帧图像
    img1_path = r"D:\Github\ITSDT\images\9\50.bmp"  # 帧 t-2
    img2_path = r"D:\Github\ITSDT\images\9\51.bmp"  # 帧 t-1
    img3_path = r"D:\Github\ITSDT\images\9\52.bmp"  # 帧 t (如果为None则使用两帧模式)
    
    # ========== 参数调整说明 ==========
    # 对于微弱运动目标，建议调整以下参数：
    # 1. motion_distance_threshold: 增大或设为None（保留所有跟踪点）
    # 2. ransac_threshold: 减小到1.0-2.0（提高精度）
    # 3. klt_epsilon: 减小到0.001-0.002（提高对微弱运动的敏感度）
    # 4. min_tracking_points: 减小到10（降低跟踪点数量要求）
    # =================================
    
    # 生成Motion Difference Map
    try:
        motion_diff_map = generate_motion_diff_map(
            img1_path=img1_path,
            img2_path=img2_path,
            img3_path=img3_path,  # 三帧模式：提供第三帧路径，设为None则使用两帧模式
            output_path=None,  # 自动生成输出路径
            show_result=True,  # 显示结果
            # 微弱运动目标优化参数（可根据实际情况调整）
            motion_distance_threshold=None,  # None表示保留所有点，或设为100-200
            min_tracking_points=10,          # 降低跟踪点数量要求
            ransac_threshold=1,           # 减小以提高精度 1
            klt_epsilon=0.1,              # 减小以提高对微弱运动的敏感度 0.1
            klt_max_iter=10,                # 增加迭代次数提高精度
            # 高频滤波参数
            use_highpass=True,              # 启用高频滤波
            highpass_d0=50,                  # 截止频率（可调整：20-50）
            highpass_n=2,                    # 滤波器阶数（可调整：1-4）
            # 显著性计算参数（三帧模式：对每个Motion Difference Map计算显著性后再融合）
            use_saliency=True,               # 启用显著性计算（三帧模式下对每个Motion Difference Map计算显著性）
            saliency_weight=0.5,              # 显著性权重（仅两帧模式使用，0-1，0.5表示等权重融合）
            # 能量累积和背景抑制参数（仅三帧模式使用）
            energy_alpha=0.8,                # 能量累积衰减系数（0-1，越大越重视新帧）
            suppression_threshold=0.2        # 背景抑制阈值（小于此值的区域被抑制）
        )
        print("\n成功生成Motion Difference Map!")
        # 处理返回值（可能是元组或单个值）
        if isinstance(motion_diff_map, tuple):
            motion_diff_map, saliency_map = motion_diff_map
            print(f"Motion Difference Map尺寸: {motion_diff_map.shape}")
            print(f"Motion Difference Map像素值范围: [{motion_diff_map.min()}, {motion_diff_map.max()}]")
            if saliency_map is not None:
                print(f"显著性图尺寸: {saliency_map.shape}")
                print(f"显著性图像素值范围: [{saliency_map.min()}, {saliency_map.max()}]")
        else:
            print(f"输出图像尺寸: {motion_diff_map.shape}")
            print(f"像素值范围: [{motion_diff_map.min()}, {motion_diff_map.max()}]")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def find_cx_cy(self) -> Tuple[int, int]:
    global up_image, up_depth, net_pose
    l=[5, 12]  # 保留节点 5 和 12
    h, w = up_image.shape[:2]
    pose = None
    poses = net_pose.forward(up_image)

    # 找到最近的人物
    min_distance = float("inf")
    closest_person = None
    reference_x, reference_y = 100, 100  # 更换为你的参考点
    for i, pose in enumerate(poses):
        # 获取人物中心点坐标
        x, y, preds = self.get_pose_target(pose, l[0])
        if preds <= 0:
            continue
        distance = self.get_real_xyz(up_depth, x, y)
        # 确保最近人物的距离不大于 1800mm
        if distance < min_distance and distance <= 1800:
            min_distance = distance
            closest_person = pose

    if closest_person is None:
        return 0, 0, up_image, "no"

    # 获取最近人物的关键点坐标
    key_points = []
    for j, num in enumerate(l):
        x, y, preds = self.get_pose_target(closest_person, num)
        if preds <= 0:
            continue
        key_points.append((x, y))

    # 计算最近人物关键点的中心点坐标
    cx, cy = np.mean(key_points, axis=0)

    # 在最近的人物周围绘制边界框
    x_min = int(np.min(key_points, axis=0)[0])
    y_min = int(np.min(key_points, axis=0)[1])
    x_max = int(np.max(key_points, axis=0)[0])
    y_max = int(np.max(key_points, axis=0)[1])
    cv2.rectangle(up_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return int(cx), int(cy), up_image, "yes"

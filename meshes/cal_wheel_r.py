import trimesh
import numpy as np

def estimate_wheel_radius(stl_path, plane='xy'):
    # 載入 mesh
    mesh = trimesh.load_mesh(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # 如果是 scene，合併
        mesh = trimesh.util.concatenate(mesh.dump())

    # 取頂點陣列
    verts = mesh.vertices  # shape (N,3)

    # 選擇要投影的平面 (預設 xy)
    if plane.lower() == 'xy':
        proj = verts[:, :2]   # x,y
    elif plane.lower() == 'xz':
        proj = verts[:, [0,2]]
    elif plane.lower() == 'yz':
        proj = verts[:, 1:]
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # 估計中心：使用投影點的幾何質心 (mean)
    center2 = proj.mean(axis=0)

    # 計算每個點到中心的距離
    dists = np.linalg.norm(proj - center2, axis=1)

    # 半徑估計：用最大距離
    radius = dists.max()

    # 也回傳外接盒參考（X/Y extents）
    bbox = mesh.bounding_box.extents

    return {
        'estimated_radius': float(radius),
        'center_2d': center2.tolist(),
        'bbox_extents': bbox.tolist(),
        'units_note': 'STL 檔案通常不含單位；結果的單位取決於該模型所用的單位（例如若原模型是以 mm 建模，則此數值為 mm）'
    }

if __name__ == "__main__":
    stl_file = "/home/ray/crazydog_mujoco/crazydog_urdf/meshes/R_wheel.STL"  # 換成你的檔案路徑
    info = estimate_wheel_radius(stl_file, plane='xy')
    print("估計半徑:", info['estimated_radius'])
    print("2D 中心:", info['center_2d'])
    print("外接盒 extents:", info['bbox_extents'])
    print(info['units_note'])

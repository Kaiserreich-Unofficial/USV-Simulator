import xml.etree.ElementTree as ET

# 缩放比例
sx, sy, sz = 1.3/4.9, 0.98/2.4, 0.98/2.4

# 读取 URDF
tree = ET.parse('src/usv_world_model/urdf/heron.urdf')
root = tree.getroot()

# 遍历所有 origin 标签
for origin in root.findall('.//origin'):
    xyz = origin.get('xyz')
    if xyz:
        x, y, z = map(float, xyz.split())
        x *= sx; y *= sy; z *= sz
        origin.set('xyz', f'{x:.6f} {y:.6f} {z:.6f}')

# （可选）遍历所有 mesh，统一在视觉/碰撞中添加 scale
for mesh in root.findall('.//geometry/mesh'):
    mesh.set('scale', f'{sx:.6f} {sy:.6f} {sz:.6f}')

# 写出新 URDF
tree.write('wamv_gazebo_scaled.urdf', encoding='utf-8', xml_declaration=True)

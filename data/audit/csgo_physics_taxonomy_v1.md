# CSGO Physics Taxonomy v1

## Primary Labels

1. `self_flying`
   - 第一人称视角本体出现悬空、漂浮、违反地面接触关系。
2. `teammate_flying`
   - 视野内玩家以飘行、悬空、无重力状态移动。
3. `static_wall_motion`
   - 墙体、建筑等静态结构在无外力条件下位移或扭曲。
4. `static_box_motion`
   - 箱子、障碍物等静态物体无原因运动或抖动。
5. `camera_teleport_or_snap`
   - 镜头突然跳变、瞬移、断裂式转向。
6. `scale_or_shape_instability`
   - 人物或场景物体出现不合理缩放、伸缩、形变。
7. `other_physics_violation`
   - 其他未覆盖但显著的物理不一致现象。

## Severity Guide

- `0`: 无异常
- `1`: 轻微，可察觉但不影响整体理解
- `2`: 明显，影响视频可信度
- `3`: 严重，核心物理规律失效

## Review Notes

- 每条样本至少记录 `clip_name`、`seed`、`frame_range`、`label`、`severity`、`note`。
- 若同一视频存在多个异常，拆成多行记录。

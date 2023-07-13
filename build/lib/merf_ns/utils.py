
from enum import Enum
class MeRFNSFieldHeadNames(Enum):
    DIFFUSE = "diffuse"
    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    NORMALS = "normals"
    PRED_NORMALS = "pred_normals"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
    SDF = "sdf"
    ALPHA = "alpha"
    GRADIENT = "gradient"
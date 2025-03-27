import os
import warnings
from typing import List, Optional, Tuple, Union
from enum import IntFlag, auto

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, create_model, Field


class ID(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    IMOD_file_id: str
    version_id: str


class GeneralStorage(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    type: int
    flags: int
    index: Union[float, int, Tuple[int, int], Tuple[int, int, int, int]]
    value: Union[float, int, Tuple[int, int], Tuple[int, int, int, int]]


class ModelFlags(IntFlag):
    flag0: bool = auto() #0
    flag1: bool = auto() #1
    flag2: bool = auto() #2
    flag3: bool = auto() #3
    flag4: bool = auto() #4
    flag5: bool = auto() #5
    flag6: bool = auto() #6
    flag7: bool = auto() #7
    flag8: bool = auto() #8
    mesh_thickness_possible: bool = auto() #9
    z_coordinates_start_from_negative_half: bool = auto() #10
    model_has_not_been_written: bool = auto() #11
    multiple_clip_planes_possible: bool = auto() #12
    mat1_and_mat3_are_bytes: bool = auto() #13
    otrans_has_image_origin_values: bool = auto() #14
    current_tilt_angles_are_stored_correctly: bool = auto() #15
    model_last_viewed_onYZ_flipped_or_rotated: bool = auto() #16
    model_rotx: bool = auto() #17


# Create pydantic model bas on the ModelFlags Flag enum
ModelFlagsModel = create_model("ModelFlagsModel", **{flag.name: (bool, Field(default=False)) for flag in ModelFlags})
    
class ModelHeader(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    name: str = 'IMOD-NewModel'
    xmax: int = 0
    ymax: int = 0
    zmax: int = 0
    objsize: int = 0
    flags: ModelFlags = ModelFlags(0)
    drawmode: int = 1
    mousemode: int = 2
    blacklevel: int = 0
    whitelevel: int = 255
    xoffset: float = 0.0
    yoffset: float = 0.0
    zoffset: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    zscale: float = 1.0
    object: int = 0
    contour: int = 0
    point: int = -1
    res: int = 3
    thresh: int = 128
    pixelsize: float = 1.0
    units: int = 0
    csum: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    @field_validator('name', mode="before")
    @classmethod
    def decode_null_terminated_byte_string(cls, value: bytes):
        end = value.find(b'\x00')
        return value[:end].decode('utf-8')

class ObjectFlags(IntFlag):
    flag0: bool = auto()
    turn_off_display: bool = auto()
    draw_using_depth_cue: bool = auto()
    open: bool = auto()                                   # 3 /* Object contains Open/Closed contours */
    wild: bool = auto()                                   # 4 /* No constraints on contour data       */
    inside_out: bool = auto()
    use_fill_for_spheres: bool = auto()
    draw_spheres_central_section_only: bool = auto()
    fill: bool = auto()
    scattered: bool = auto()
    mesh: bool = auto()                                   # 10 /* Draw mesh in 3D, imod view        */
    noline: bool = auto()                                 # 11 
    use_value: bool = auto()                              # 12 
    planar: bool = auto()                                 # 13 
    fcolor: bool = auto()                                 # 14 
    anti_alias: bool = auto()                             # 15 
    scalar: bool = auto()                                 # 16 
    mcolor: bool = auto()                                 # 17 
    time: bool = auto()                                   # 18 
    two_side: bool = auto()                               # 19 
    thick_cont: bool = auto()                             # 20 
    extra_modv: bool = auto()                             # 21 
    extra_edit: bool = auto()                             # 22 
    pnt_nomodv: bool = auto()                             # 23 
    modv_only: bool = auto()                              # 24 
    flag25: bool = auto()                                 # 25 
    poly_cont: bool = auto()                              # 26 
    draw_label: bool = auto()                             # 27 
    scale_wdth: bool = auto()                             # 28 


ObjectFlagsModel = create_model("ObjectFlagsModel", **{flag.name: (bool, Field(default=False)) for flag in ObjectFlags})


class ObjectHeader(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    name: str = ''
    extra_data: List[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    contsize: int = 0
    flags: ObjectFlags = ObjectFlags(0)
    axis: int = 0
    drawmode: int = 1
    red: float = 0.0
    green: float = 1.0
    blue: float = 0.0
    pdrawsize: int = 2
    symbol: int = 1
    symsize: int = 3
    linewidth2: int = 1
    linewidth: int = 1
    linesty: int = 0
    symflags: int = 0
    sympad: int = 0
    trans: int = 0
    meshsize: int = 0
    surfsize: int = 0

    @field_validator('name', mode="before")
    @classmethod
    def decode_null_terminated_byte_string(cls, value: bytes):
        end = value.find(b'\x00')
        return value[:end].decode('utf-8')

class ContourFlags(IntFlag):
    flag0: bool = auto()
    flag1: bool = auto()
    flag2: bool = auto()
    open: bool = auto()
    wild: bool = auto()
    strippled: bool = auto()
    cursor_like: bool = auto()
    draw_allz: bool = auto()
    mmodel_only: bool = auto()
    noconnect: bool = auto()
    flag10: bool = auto()
    flag11: bool = auto()
    flag12: bool = auto()
    flag13: bool = auto()
    flag14: bool = auto()
    flag15: bool = auto()
    flag16: bool = auto()
    scanline: bool = auto()
    connect_top: bool = auto()
    connect_bottom: bool = auto()
    connect_invert: bool = auto()
    

ContourFlagsModel = create_model("ContourFlagsModel", **{flag.name: (bool, Field(default=False)) for flag in ContourFlags})


class ContourHeader(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    psize: int = 0
    flags: ContourFlags = ContourFlags(0)
    time: int = 0
    surf: int = 0


class Contour(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    header: ContourHeader = ContourHeader()
    points: np.ndarray  # pt
    point_sizes: Optional[np.ndarray]  = None
    extra: List[GeneralStorage] = []

    model_config = ConfigDict(arbitrary_types_allowed=True,
                              validate_assignment=True)
        
    @model_validator(mode='after')
    def update_sizes(self):
        self.header.psize = len(self.points)
        return(self)


class MeshHeader(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    vsize: int
    lsize: int
    flag: int
    time: int
    surf: int

class Mesh(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    header: MeshHeader
    raw_vertices: np.ndarray
    raw_indices: np.ndarray
    extra: List[GeneralStorage] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('raw_indices')
    @classmethod
    def validate_indices(cls, indices: np.ndarray):
        if indices.ndim > 1:
            raise ValueError('indices must be 1D')
        if indices[-1] != -1:
            raise ValueError('Indices must end with -1')
        if len(indices[np.where(indices >= 0)]) % 3 != 0:
            raise ValueError(f'Invalid indices shape: {indices.shape}')
        for i in (-20, -23, -24):
            if i in indices:
                warnings.warn(f'Unsupported mesh type: {i}')
        return indices

    @field_validator('raw_vertices')
    @classmethod
    def validate_vertices(cls, vertices: np.ndarray):
        if vertices.ndim > 1:
            raise ValueError('vertices must be 1D')
        if len(vertices) % 3 != 0:
            raise ValueError(f'Invalid vertices shape: {vertices.shape}')
        return vertices

    @property
    def vertices(self) -> np.ndarray:
        return self.raw_vertices.reshape((-1, 3))

    @property
    def indices(self) -> np.ndarray:
        return self.raw_indices[np.where(self.raw_indices >= 0)].reshape((-1, 3))

    @property
    def face_values(self) -> Optional[np.ndarray]:
        """Extra value for each vertex face.
        The extra values are index, value  pairs
        However, the index is an index into the indices array,
        not directly an index of a vertex.
        Furthermore, the index has to be fixed because
        the original indices array has special command values (-25, -22, -1, ...)
        """
        values = np.zeros((len(self.vertices),))
        has_face_values = False
        for extra in self.extra:
            if not (extra.type == 10 and isinstance(extra.index, int)):
                continue
            has_face_values = True
            values[self.raw_indices[extra.index]] = extra.value
        if has_face_values:
            return values


class IMAT(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    ambient: int = 102
    diffuse: int = 255
    specular: int = 127
    shininess: int = 4
    fillred: int = 0
    fillgreen: int = 0
    fillblue: int = 0
    quality: int = 0
    mat2: int = 0
    valblack: int = 0
    valwhite: int = 255
    matflags2: int = 0 
    mat3b3: int = 0

class MINX(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    oscale: Tuple[float, float, float]
    otrans: Tuple[float, float, float]
    orot: Tuple[float, float, float]
    cscale: Tuple[float, float, float]
    ctrans: Tuple[float, float, float]
    crot: Tuple[float, float, float]

class View(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    fovy: float
    rad: float
    aspect: float
    cnear: float
    cfar: float
    rot: Tuple[float, float, float]
    trans: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    mat: Tuple[
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
    ]
    world: int
    label: str
    dcstart: float
    dcend: float
    lightx: float
    lighty: float
    plax: float
    objvsize: int
    bytesObjv: int


class SLAN(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    time: int
    angles: Tuple[float,float,float]
    center: Tuple[float,float,float]
    label: bytes


class Object(BaseModel):
    """https://bio3d.colorado.edu/imod/doc/binspec.html"""
    header: ObjectHeader = ObjectHeader()
    contours: Tuple[Contour,...] = ()
    meshes: Tuple[Mesh,...] = ()
    extra: List[GeneralStorage] = []
    imat: Optional[IMAT] = None
    cview: Optional[int] = None

    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, 
                 color: Optional[Tuple[float, float, float]] = None, 
                 flags: Optional[ObjectFlagsModel] = None,
                 **data):
        super().__init__(**data)
        if color is not None:
            self.color = color
        if flags is not None:
            self.flags = flags
    
    @model_validator(mode='after')
    def update_sizes(self):
        self.header.contsize = len(self.contours)
        self.header.meshsize = len(self.meshes)
        return(self)
    
    @property
    def color(self):
        return (self.header.red, self.header.green, self.header.blue)

    @color.setter
    def color(self, value: Tuple[float, float, float]):
        self.header.red, self.header.green, self.header.blue = value

    @property
    def flags(self) -> ObjectFlagsModel:
        return ObjectFlagsModel(**{flag.name: True for flag in self.header.flags})
    
    @flags.setter
    def flags(self, model: ObjectFlagsModel):
        for name, value in model:
            if value:
                self.header.flags |= ObjectFlags[name]
            else:
                self.header.flags &= ~ObjectFlags[name]

class ImodModel(BaseModel):
    """Contents of an IMOD model file.

    https://bio3d.colorado.edu/imod/doc/binspec.html
    """
    id: ID = ID(IMOD_file_id='IMOD', version_id='V1.2')
    header: ModelHeader = ModelHeader()
    objects: Tuple[Object,...] = []
    slicer_angles: List[SLAN] = []
    minx: Optional[MINX] = None
    extra: List[GeneralStorage] = []
    flags: ModelFlagsModel = ModelFlagsModel()

    model_config = ConfigDict(validate_assignment=True,
                              arbitrary_types_allowed=True)
    
    @model_validator(mode='after')
    def update_sizes(self):
        self.header.objsize = len(self.objects)
        return(self)
    
    @classmethod
    def from_file(cls, filename: os.PathLike):
        """Read an IMOD model from disk."""
        from .parsers import parse_model
        with open(filename, 'rb') as file:
            return parse_model(file)
    
    def to_file(self, filename: os.PathLike):
        """Write an IMOD model to disk."""
        from .writers import write_model
        with open(filename, 'wb') as file:
            write_model(file, self)

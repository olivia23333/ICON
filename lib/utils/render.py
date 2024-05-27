import cv2
import torch
import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures

class Renderer():
    def __init__(self, image_size=256, anti_alias=False):
        super().__init__()

        self.anti_alias = anti_alias

        self.image_size = image_size

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # R = torch.from_numpy(np.array([[-1., 0., 0.],
        #                                [0., 1., 0.],
        #                                [0., 0., -1.]])).float().unsqueeze(0).to(self.device)


        # t = torch.from_numpy(np.array([[0., 0., 2.]])).float().to(self.device)
        # self.R = R

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        if anti_alias: image_size = image_size*2
        self.raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=100, blur_radius=0)
    
    # def prepare(self, c2ws, c2w_init, real_cam=True):
    def prepare(self, c2ws, real_cam=True):
        c2ws[:,:2] *= (-1)
        # c2w_init[:,:2] *= (-1)
        if real_cam:
            # self.cameras = FoVPerspectiveCameras(fov=11.69, R=c2ws[:3,:3][None], T=c2w_init[:3, 3][None], device=self.device)
            self.cameras = FoVPerspectiveCameras(fov=11.69, R=c2ws[:3,:3][None], T=np.array([0,0,10])[None], device=self.device)
        else:
            self.cameras = FoVOrthographicCameras(R=c2ws[:3,:3][None], T=c2w_init[:3, 3][None], device=self.device)

        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh(self, mesh):
        with torch.no_grad():
            image_color = self.renderer(mesh)[0]
            if self.anti_alias:
                image_color = image_color.permute(2, 0, 1)  # NHWC -> NCHW
                image_color = torch.nn.functional.interpolate(image_color, scale_factor=0.5,mode='bilinear',align_corners=True)
                image_color = image_color.permute(1, 2, 0)  # NCHW -> NHWC                    
            image_color = (255*image_color).data.cpu().numpy().astype(np.uint8)
            return image_color
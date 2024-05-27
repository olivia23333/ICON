import argparse
import os, sys
import cv2
import trimesh
import numpy as np
import random
import math
import random
from tqdm import tqdm
import torch
import json
from PIL import Image

# multi-thread
# from functools import partial
# from multiprocessing import Pool, Queue
# import multiprocessing as mp

# to remove warning from numba
# "Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.""
# import numba
# numba.config.THREADING_LAYER = 'workqueue'

sys.path.append(os.path.join(os.getcwd()))

def generate_cameras(dist=10, view_num=60):
    cams = []
    v_view = [math.pi/2 - 0.4, math.pi/2, math.pi/2 + 0.4]
    target = [0, 0, 0]
    up = [0, 1, 0]
    angles_vis = []
    for v_angle in v_view:
        for view_idx in range(view_num):
            angle = (math.pi * 2 / view_num) * view_idx
            angle_vis = int((360 / view_num) * view_idx)
            eye = np.asarray([dist * math.sin(angle) * math.sin(v_angle), dist * math.cos(v_angle), dist * math.cos(angle) * math.sin(v_angle)])

            fwd = np.asarray(target, np.float64) - eye
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)

            cams.append(
                {
                    'center': eye, 
                    'direction': fwd, 
                    'right': right, 
                    'up': -down, 
                }
            )
            angles_vis.append(angle_vis)

    return cams, angles_vis

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def render_sides(render_types, rndr, y, save_folder, subject, smpl_type, side):

    if "normal" in render_types:
        opengl_util.render_result(
            rndr, 1, os.path.join(save_folder, subject, f"normal_{side}", f'{y:03d}.png')
        )

    if "depth" in render_types:
        opengl_util.render_result(
            rndr, 2, os.path.join(save_folder, subject, f"depth_{side}", f'{y:03d}.png')
        )


def render_subject(subject, path, dataset, save_folder, rotation, size, render_types, rndr, color_rndr, egl, ortho):

    scale = 1.0
    up_axis = 1
    smpl_type = "smplx"

    if dataset == '2K2K':
        mesh_file = os.path.join(path, subject, subject + '.ply')
    elif dataset == 'THuman':
        mesh_file = os.path.join(path, subject, subject + '.obj')
        tex_file = os.path.join(path, subject, 'material0.jpeg')
        # fit_file = os.path.join(os.path.split(path)[0], 'THuman2.0_smpl', subject+'_smpl.pkl')
        fit_file = os.path.join(os.path.split(path)[0], 'THUman20_Smpl-X', subject, 'smplx_param.pkl')
    elif dataset == 'Custom':
        mesh_file = os.path.join(path, subject, 'mesh-f' + subject[-5:] + '.obj')
        tex_file = os.path.join(path, subject, 'mesh-f' + subject[-5:] + '.png')
        fit_file = os.path.join(os.path.split(path)[0], 'smplx', subject, 'mesh-f' + subject[-5:] + '.json')
    elif dataset == 'faceverse':
        mesh_file = os.path.join(path, subject, subject + '.obj')
        tex_file = os.path.join(path, subject, subject + '.jpg')
        fit_file = os.path.join(os.path.split(path)[0], 'flame', 'flame_params',  subject + '_flame.pkl')
    elif dataset == 'sizer':
        mesh_file = os.path.join(path, subject, 'model_0.8.obj')
        tex_file = os.path.join(path, subject, 'model_0.8.jpg')
        # fit_file = os.path.join()
    elif dataset == 'SynBody':
        mesh_file = os.path.join(path, subject, 'SMPL-XL-Tpose.obj')
        tex_file = os.path.join(path, subject, 'SMPL-XL-Tpose.mtl')
    elif dataset == 'cloth':
        mesh_file = os.path.join(path, subject, subject+'.npz')
    else:
        assert False
    
    if dataset == 'SynBody':
        scene = trimesh.load_mesh(mesh_file, process=False)
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
        mesh = trimesh.load_mesh(mesh_file, process=False)
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        faces_normals = mesh.face_normals
        textures = mesh.visual.vertex_colors
        face_textures = mesh.visual.face_colors
        
    elif mesh_file[-3:] == 'obj':
        vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )
    elif mesh_file[-3:] == 'ply':
        try:
            mesh = trimesh.load_mesh(mesh_file, process=True)
            vertices = mesh.vertices
            faces = mesh.faces
            normals = mesh.vertex_normals
            faces_normals = mesh.face_normals
            textures = mesh.visual.vertex_colors
            face_textures = mesh.visual.face_colors
        except:
            print(mesh_file)
            return
    elif mesh_file[-3:] == 'npz':
        mesh_attr = np.load(mesh_file)
        vertices = mesh_attr['points']
        faces = mesh_attr['faces']
        normals = mesh_attr['normals']
        colors = mesh_attr['colors']
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=colors)
        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        textures = mesh.visual.vertex_colors
    else:
        assert False

    # center
    scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    if mesh_file[-3:] == 'ply' or mesh_file[-3:]=='npz' or dataset == 'SynBody':
        pass
    elif dataset == 'Custom':
        fit_param, rescale_fitted_body, joints, transl = load_fit_body(
            fit_file, scan_scale, smpl_type=smpl_type, smpl_gender='male', dataset='Custom'
        )
    elif dataset == 'faceverse':
        fit_param, flame_model, flame_mesh = load_fit_face(fit_file)
    else:
        if smpl_type == 'smplx':
            fit_param = load_fit_body(
                fit_file, scan_scale, smpl_type=smpl_type, smpl_gender='male', dataset='THuman'
            )
        else:
            fit_param, rescale_fitted_body, joints = load_fit_body(
                fit_file, scan_scale, smpl_type=smpl_type, smpl_gender='neutral', dataset='THuman'
            )

    if dataset == 'faceverse':
        # flip and translate to origin
        vertices[:,1] = -vertices[:,1]
        vertices[:,2] = -vertices[:,2]
        t = (vertices[:,2].max() + vertices[:,2].min()) / 2
        vertices[:,2] = vertices[:,2] - t
        # add scale and transl to align with flame model
        flame_scale = np.array(fit_param['scale'])
        flame_transl = np.array(fit_param['transl'])
        flame_t = np.zeros_like(flame_transl)    
        flame_t[:,2] = flame_model.t
        scan_scale = flame_model.factor / flame_scale
        vertices = scan_scale * (vertices - flame_transl) + flame_t
    elif dataset == 'SynBody':
        smpl_t = np.array([0, 0, 0])[None]
        smpl_t[:, 1] = 0.25    
        vertices -= np.array([0, 1.15, 0])[None]
        vertices = vertices / 1. + smpl_t
    elif dataset == 'THuman':
        # smpl_t = np.zeros_like(fit_param['transl'])
        smpl_t = np.zeros_like(fit_param['translation'])[None]
        smpl_t[:, 1] = 0.35
        # vertices -= np.array(fit_param['transl'])
        vertices -= np.array(fit_param['translation'])
        vertices = vertices / np.array(fit_param['scale'][0]) + smpl_t
    elif dataset == 'Custom':
        smpl_t = np.zeros_like(fit_param['transl'])[None]
        smpl_t[:, 1] = 0.1 
        vertices = vertices / 1. + smpl_t
    elif dataset == '2K2K':
        vert_med = vertices.mean(0)
        vertices -= vert_med
        smpl_t = np.array([0, 0, 0])[None]
    else:
        smpl_t = np.array([0, 0, 0])[None]


    cam = Camera(width=size, height=size, focal=5000, near=0.1, far=40)
    cam.sanity_check()

    if mesh_file[-3:] == 'ply' or dataset == 'SynBody':
        texture = textures / 255.
        color_rndr.set_mesh(vertices, faces, texture[:,:3], normals)
        # color_rndr.set_norm_mat(scan_scale, vmed)
        color_rndr.set_norm_mat(1.0, 0.0)
    elif mesh_file[-3:] == 'npz':
        color_rndr.set_mesh(vertices, faces, colors, normals)
        color_rndr.set_norm_mat(1.0, 0.0)
    elif mesh_file[-3:] == 'obj':
        prt, face_prt = prt_util.computePRT(mesh_file, scan_scale, 10, 2)
        texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)
        tan, bitan = compute_tangent(normals)
        rndr.set_norm_mat(1.0, 0.0)
        
        rndr.set_mesh(
            vertices,
            faces,
            normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
            np.zeros((vertices.shape[0], 3)),
        )
        rndr.set_albedo(texture_image)
        
        if args.debug:
            model_mesh = trimesh.Trimesh(vertices, faces, process=False, maintain_order=True)
            if dataset == 'faceverse':
                rndr_smpl.set_mesh(flame_mesh.vertices, flame_mesh.faces, flame_mesh.vertex_normals, \
                    flame_mesh.vertex_normals)
                rndr_smpl.set_norm_mat(1.0, 0.0)
                # flame_mesh.export(os.path.join(save_folder, 'render', 'flame_mesh.obj'))
            else:
                if dataset == 'THuman':
                    # smpl_mesh = get_smpl(fit_file, 1.0, smpl_t, smpl_type=smpl_type, smpl_gender='neutral', dataset='THuman')
                    smpl_mesh = get_smpl(fit_file, 1.0, smpl_t, smpl_type=smpl_type, smpl_gender='male', dataset='THuman')
                elif dataset == 'Custom':
                    smpl_mesh = get_smpl(fit_file, 1.0, smpl_t, smpl_type=smpl_type, smpl_gender='male')
                else:
                    assert False
                rndr_smpl.set_mesh(smpl_mesh.vertices, smpl_mesh.faces, smpl_mesh.vertex_normals*0.5+0.5, \
                        smpl_mesh.vertex_normals)
                rndr_smpl.set_norm_mat(1.0, 0.0)
    else:
        assert False

    cam_params, angles = generate_cameras(dist=10, view_num=rotation)
    
    for i, (cam_param, angle) in enumerate(zip(cam_params, angles)):

        # cam.near = -100
        # cam.far = 100
        cam.far = 40
        cam.near = 0.1
      
        cam.center = cam_param['center']
        cam.right = cam_param['right']
        cam.up = cam_param['up']
        cam.direction = cam_param['direction']
        cam.sanity_check()
        extrinsic = cam.get_extrinsic_matrix()
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(np.array(extrinsic))

        if mesh_file[-3:] == 'obj':
            rndr.set_camera(cam)
            if args.debug:
                rndr_smpl.set_camera(cam)
        elif mesh_file[-3:] == 'ply':
            color_rndr.set_camera(cam)
        elif mesh_file[-3:] == 'npz':
            color_rndr.set_camera(cam)
        else:
            assert False

        if dataset == 'faceverse':
            dic = {'scale': 1.0, 'center': [0, 0, 0], 'cam_param': c2w.tolist()}
        elif dataset == 'Custom':
            dic = {'scale': 1.0, 'center': (transl[0]+smpl_t[0]).tolist(), 'cam_param': c2w.tolist()}
        else:
            dic = {'scale': 1.0, 'center': smpl_t[0].tolist(), 'cam_param': c2w.tolist()}
        
        if "light" in render_types:

            # random light
            shs = np.load('./scripts/env_sh.npy')
            sh_id = random.randint(0, shs.shape[0] - 1)
            sh = shs[sh_id]
            sh_angle = 0.2 * np.pi * (random.random() - 0.5)
            sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
            dic.update({"sh": sh})

            if mesh_file[-3:] == 'obj':
                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False
            elif mesh_file[-3:] == 'ply':
                color_rndr.set_sh(sh)
                color_rndr.analytic = False
                color_rndr.use_inverse_depth = False
            elif mesh_file[-3:] == 'npz':
                color_rndr.set_sh(sh)
                color_rndr.analytic = False
                color_rndr.use_inverse_depth = False
            else:
                assert False

        if mesh_file[-3:] == 'ply' or mesh_file[-3:] == 'npz' or dataset=='SynBody':
            shs = np.load('./scripts/env_sh.npy')
            sh_id = 0
            sh = shs[sh_id]
            sh_angle = 0.2 * np.pi * (random.random() - 0.5)
            sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
            dic.update({"sh": sh.tolist()})
            color_rndr.set_sh(sh)
            color_rndr.analytic = False
            color_rndr.use_inverse_depth = False
        # ==================================================================

        export_calib_file = os.path.join(save_folder, 'calib', f'{angle:03d}_{(i//rotation):03d}.json')
        os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
        with open(export_calib_file, 'w') as f:
            json.dump(dic, f)
        
        # np.savetxt(export_calib_file, calib_info)

        # ==================================================================

        # render
        if mesh_file[-3:] == 'ply' or mesh_file[-3:] == 'npz' or dataset=='SynBody':
            color_rndr.display()
            opengl_util.render_result(
                color_rndr, 0, os.path.join(save_folder, 'render', f'{angle:03d}_{(i//rotation):03d}.png')
            )
        elif mesh_file[-3:] == 'obj':
            rndr.display()
            opengl_util.render_result(
                rndr, 0, os.path.join(save_folder, 'render', f'{angle:03d}_{(i//rotation):03d}.png')
            )
            if args.debug:
                rndr_smpl.display()
                opengl_util.render_result(
                    rndr_smpl, 0, os.path.join(save_folder, 'render', f'{angle:03d}_{(i//rotation):03d}_smpl.png')
                )
                model_mesh.export(os.path.join(save_folder, 'render', 'mesh.obj'))
                if dataset == 'faceverse':
                    flame_mesh.export(os.path.join(save_folder, 'render', 'flame_mesh.obj'))
                else:
                    smpl_mesh.export(os.path.join(save_folder, 'render', 'smpl_mesh.obj'))
        else:
            assert False

    if mesh_file[-3:] == 'ply' or mesh_file[-3:] == 'npz' or dataset=='SynBody':
        color_rndr.cleanup()
    elif mesh_file[-3:] == 'obj':
        rndr.cleanup()
        if args.debug:
            rndr_smpl.cleanup()
    else:
        assert False
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman", help='dataset name')
    parser.add_argument('-path', '--path', type=str, default="HumanData", help='dataset path')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./debug", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=18, help='number of views')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')
    parser.add_argument(
        '-debug', '--debug', action="store_true", help='debug mode, only render one subject'
    )
    parser.add_argument(
        '-headless', '--headless', action="store_true", help='headless rendering with EGL'
    )
    args = parser.parse_args()

    # rendering setup
    if args.headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ["PYOPENGL_PLATFORM"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # shoud be put after PYOPENGL_PLATFORM
    import lib.renderer.opengl_util as opengl_util
    from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent, get_smpl, load_fit_face
    import lib.renderer.prt_util as prt_util
    from lib.renderer.gl.init_gl import initialize_GL_context
    from lib.renderer.gl.prt_render import PRTRender
    from lib.renderer.gl.color_render import ColorRender
    from lib.renderer.camera import Camera

    print(
        f"Start Rendering {args.dataset} with {args.num_views * 3} views, {args.size}x{args.size} size."
    )

    subjects = os.listdir(args.path)

    if args.debug:
        subjects.sort()
        subjects = subjects[:3]
        print(subjects)
        render_types = ["normal", "depth"]
        # render_types = ["light", "normal", "depth"]
    else:
        # random.shuffle(subjects)
        subjects.sort()
        render_types = ["normal", "depth"]

    print(f"Rendering types: {render_types}")

    # setup global rendering parameter
    initialize_GL_context(width=args.size, height=args.size, egl=args.headless)

    if args.dataset == '2K2K' or args.dataset == 'SynBody' or args.dataset == 'cloth':
        color_rndr = ColorRender(width=args.size, height=args.size, egl=args.headless)
        rndr = None
        rndr_smpl = None
    else:
        rndr = PRTRender(width=args.size, height=args.size, ms_rate=4, egl=args.headless)
        color_rndr = None
        if args.debug:
            rndr_smpl = ColorRender(width=args.size, height=args.size, egl=args.headless)

    for subject in tqdm(subjects):
        if not args.debug:
            current_out_dir = f"{args.path}/{subject}/{args.num_views}views_3"
        else:
            current_out_dir = f"./debug/{subject}"
        os.makedirs(current_out_dir, exist_ok=True)
        print(f"Output dir: {current_out_dir}")
        render_subject(
                    subject,
                    path=args.path,
                    dataset=args.dataset,
                    save_folder=current_out_dir,
                    rotation=args.num_views,
                    size=args.size,
                    egl=args.headless,
                    render_types=render_types,
                    rndr=rndr,
                    color_rndr=color_rndr,
                    ortho=True)

    print('Finish Rendering.')

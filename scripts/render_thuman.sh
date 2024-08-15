export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3
CUDA_VISIBLE_DEVICES=0 python scripts/render_batch.py -headless --dataset 'THuman' -path '../data/THuman/THuman2.0_Release' --size 1024 --debug
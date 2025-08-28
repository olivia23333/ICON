export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3
CUDA_VISIBLE_DEVICES=0 python scripts/render_batch.py -headless --dataset 'THuman21' -path data/THuman2_1/model --size 1024
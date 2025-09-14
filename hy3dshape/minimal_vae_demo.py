# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from pathlib import Path
import torch

from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh

##### config path: '/home/ubuntu/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/config.yaml' ####

# num_points = [81920, 10240]
num_points = [(10240, 4096)]
prefix = Path('/data/grabcad_meshed_data/grabcad/meshes/')

keys = [
    "VictorD~-a-simple-vertical-steam-engine-with-slide-crank-scotch-yoke-type-1~-FULL ASSEMBLY~-0",
    "Utkarsh Patel~-engine-blower-assembly-11~-Engine Blower Assembly~-0",
    "Ahmet Sait ÇÜNGÜRLÜ~-v6-motor-7~-V6 MOTOR Assembly~-0",
    "Brian Lee~-ninjineer-2383-star-drive-module-mk2-1~-Star Drive MK2 v21~-0",
    "Thành Đức Machine CNC~-3d-model-of-parallel-transfer-structure-after-180-flip-1~-1100-4#~-0",
]


vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    use_safetensors=False,
    variant='fp16',
    # pc_sharpedge_size=1024,
)

for num_p in num_points:
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=num_p[1],
        num_uniform_points=num_p[0],
    )
    
    for k in keys:
        print(f'Processing {k} with {num_p} points')
        mesh_path = (prefix / k.rsplit('~-', 1)[0] / k).with_suffix(".stl") 
        
        if not mesh_path.exists():
            print(f'File {mesh_path} does not exist, skipping.')
            continue

        surface = loader(str(mesh_path)).to('cuda', dtype=torch.float16)
        print(surface.shape)

        latents = vae.encode(surface)
        latents = latents.latent_dist.mode()
        latents = vae.decode(latents)
        mesh = vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=1.01,
            mc_level=0.0,
            num_chunks=20000,
            octree_resolution=256,
            mc_algo='mc',
            enable_pbar=True
        )

        mesh = export_to_trimesh(mesh)[0]
        mesh.export(f'/home/ubuntu/Code/LeoCAD/script_visualization/{k}__{num_p[0]}__edge_{num_p[1]}.glb')

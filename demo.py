#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r_896L.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import json
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma
from tqdm import tqdm

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--download_if_missing",
        action="store_true",
        help="モデルファイルが見つからない場合に Hugging Face からダウンロードします。",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="faneggg/human3r",
        help="自動ダウンロードに使う Hugging Face の repo ID。",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Hugging Face のアクセストークン。空なら HUGGINGFACE_HUB_TOKEN または HF_TOKEN を使います。",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--mat5_dir",
        type=str,
        default="../mmd-auto-trace-5",
        help="Path to mmd-auto-trace-5 directory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output results.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="関節の3D位置のみをJSONで保存し、可視化や他の出力を無視します。",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save smpl mesh projection.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Save smpl mesh projection video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--block_frame_num",
        type=int,
        default=None,
        help="ブロックごとのフレーム数。block_index*block_frame_num から切り出します。",
    )
    parser.add_argument(
        "--block_index",
        type=int,
        default=0,
        help="ブロック番号 (0始まり)。block_frame_num 指定時のみ使用します。",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    return parser.parse_args()


def _get_hf_token(cli_token):
    if cli_token:
        return cli_token
    env_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    return env_token


def _resolve_model_path(args):
    model_path = args.model_path
    if os.path.exists(model_path):
        return model_path

    if not args.download_if_missing:
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {model_path}"
        )

    if not model_path.endswith((".pth", ".pt", ".bin")):
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {model_path}"
        )

    filename = os.path.basename(model_path)
    local_dir = os.path.dirname(model_path) or "."
    token = _get_hf_token(args.hf_token)

    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError as e:
        raise ImportError(
            "huggingface_hub が見つかりません。requirements を確認してください。"
        ) from e

    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    return hf_hub_download(
        repo_id=args.hf_repo,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from src.dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in tqdm(range(len(images)), desc="Preparing input views"):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in tqdm(range(num_views), desc="Preparing input views"):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True,
        save=False, render=False, render_video=False, img_res=None, subsample=1, save_json=False,
        frame_offset=0, json_frame_tag=None):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save (bool): Whether to save output results.
        render (bool): Whether to save smpl mesh projection.
        render_video (bool): Whether to save smpl mesh projection video.
        frame_offset (int): JSON に書き出すフレーム番号のオフセット。
        json_frame_tag (str|None): JSON ファイル名に付ける開始フレーム表記 (例: "00000")。
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer
    from src.dust3r.utils.image import unpad_image
    if render:
        from src.dust3r.utils import vis_heatmap, render_meshes
        from viser_utils import get_color
    json_only = save_json and not (save or render or render_video)

    print("prepare_output: 01")
    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    print("prepare_output: 02")
    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    need_viewer = not save and not save_json
    print("prepare_output: 03")
    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    pts3ds_other = (
        [output["pts3d_in_other_view"] for output in outputs["pred"]] if need_viewer else []
    )
    conf_other = [output["conf"] for output in outputs["pred"]] if need_viewer else []
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    print("prepare_output: 04")
    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    print("prepare_output: 05")
    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    if json_only:
        for pred in outputs["pred"]:
            pred.pop("pts3d_in_self_view", None)
            pred.pop("pts3d_in_other_view", None)
            pred.pop("conf_self", None)
            pred.pop("conf", None)
        outputs["views"] = []

        if not outputs["pred"]:
            return ([], [], [], {}, [], None, [], [])

        first_shape = outputs["pred"][0].get("smpl_shape", torch.empty(1, 0, 10))[0]
        num_betas = first_shape.shape[-1] if first_shape.ndim >= 2 else 10
        smpl_layer = SMPL_Layer(
            type="smplx",
            gender="neutral",
            num_betas=num_betas,
            kid=False,
            person_center="head",
        )
        joint_names = smpl_layer.joint_names
        joints_json_by_human = {}
        axis_sign = {"x": 1.0, "y": -1.0, "z": 1.0}

        def joints_to_dict(joints, names):
            return {
                name: {
                    "x": float(joint[0] * axis_sign["x"]),
                    "y": float(joint[1] * axis_sign["y"]),
                    "z": float(joint[2] * axis_sign["z"]),
                }
                for name, joint in zip(names, joints)
            }

        def normalize_human_index(human_id):
            try:
                return int(human_id)
            except (TypeError, ValueError):
                return str(human_id)

        def get_human_entry(human_id):
            human_key = str(human_id)
            if human_key not in joints_json_by_human:
                joints_json_by_human[human_key] = {
                    "human_index": normalize_human_index(human_id),
                    "frames": {},
                }
            return human_key

        os.makedirs(os.path.join(outdir, "json"), exist_ok=True)
        json_suffix = f"_{json_frame_tag}" if json_frame_tag is not None else ""
        for f_id in tqdm(range(len(outputs["pred"])), desc="Processing frames"):
            pred = outputs["pred"][f_id]
            smpl_shape = pred.get("smpl_shape", torch.empty(1, 0, 10))[0]
            n_humans_i = smpl_shape.shape[0]
            if n_humans_i <= 0:
                continue

            smpl_rotmat = pred.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]
            smpl_rotvec = roma.rotmat_to_rotvec(smpl_rotmat)
            smpl_transl = pred.get("smpl_transl", torch.empty(1, 0, 3))[0]
            smpl_expression = pred.get("smpl_expression", [None])[0]
            smpl_id = pred.get("smpl_id", torch.empty(1, 0))[0]

            intrinsics = torch.eye(3, device=smpl_shape.device).unsqueeze(0).repeat(
                n_humans_i, 1, 1
            )
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec,
                    smpl_shape,
                    smpl_transl,
                    None,
                    None,
                    K=intrinsics,
                    expression=smpl_expression,
                )

            j3d_cam = smpl_out["smpl_j3d"].detach().cpu().numpy()
            j3d_world = (
                geotrf(pr_poses[f_id], smpl_out["smpl_j3d"].unsqueeze(0))[0]
                .detach()
                .cpu()
                .numpy()
            )
            names = joint_names[: j3d_cam.shape[1]]
            person_ids = (
                smpl_id.detach().cpu().numpy().tolist()
                if smpl_id.numel() > 0
                else list(range(n_humans_i))
            )
            for idx, pid in enumerate(person_ids[: j3d_cam.shape[0]]):
                human_key = get_human_entry(pid)
                joints_json_by_human[human_key]["frames"][str(f_id + frame_offset)] = {
                    "3d_joints": joints_to_dict(j3d_cam[idx], names),
                    "global_3d_joints": joints_to_dict(j3d_world[idx], names),
                }

        json_dir = os.path.join(outdir, "json")
        for human_key, data in joints_json_by_human.items():
            json_path = os.path.join(
                json_dir,
                f"joints_3d_human_{human_key}_original{json_suffix}.json",
            )
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

        return ([], [], [], {}, [], None, [], [])

    print("prepare_output: 06")
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    print("prepare_output: 07")
    if use_pose and need_viewer:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    print("prepare_output: 08")
    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    print("prepare_output: 09")
    if need_viewer:
        colors = [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
        ]
    else:
        colors = []

    print("prepare_output: 10")
    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    print("prepare_output: 11")
    depths_tosave = None
    conf_self_tosave = None
    colors_tosave = None
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]
    if save:
        depths_tosave = pts3ds_self[..., 2]
        conf_self_tosave = torch.cat(conf_self)  # B, H, W
    if save or render:
        colors_tosave = torch.cat(
            [
                0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
                for output in outputs["views"]
            ]
        )  # [B, H, W, 3]

    print("prepare_output: 12")
    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    # smpl_loc = [output.get(
    #     "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
    # K_mhmr = [output.get(
    #     "K_mhmr", torch.empty(1,0,3))[0] for output in outputs["views"]]
        
    print("prepare_output: 13")
    if render or save:
        smpl_scores = [
            output.get("smpl_scores", torch.zeros(1, H, W, 1))[...,0] for output in outputs["pred"]]
        if img_res is not None:
            smpl_scores = [
                unpad_image(s, [H, W])[0] for s in smpl_scores]

    print("prepare_output: 14")
    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    print("prepare_output: 15")
    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces
    joint_names = smpl_layer.joint_names

    joints_json_by_human = {} if (save or save_json) else None
    axis_sign = {"x": 1.0, "y": -1.0, "z": 1.0}

    print("prepare_output: 16")
    def joints_to_dict(joints, names):
        return {
            name: {
                "x": float(joint[0] * axis_sign["x"]),
                "y": float(joint[1] * axis_sign["y"]),
                "z": float(joint[2] * axis_sign["z"]),
            }
            for name, joint in zip(names, joints)
        }

    def normalize_human_index(human_id):
        try:
            return int(human_id)
        except (TypeError, ValueError):
            return str(human_id)

    def get_human_entry(human_id):
        human_key = str(human_id)
        if human_key not in joints_json_by_human:
            joints_json_by_human[human_key] = {
                "human_index": normalize_human_index(human_id),
                "frames": {},
            }
        return human_key

    print("prepare_output: 17")
    if save:
        print(f"Saving output to {outdir}...")
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "json"), exist_ok=True)
    elif save_json:
        os.makedirs(os.path.join(outdir, "json"), exist_ok=True)

    print("prepare_output: 18")
    all_verts = []
    for f_id in tqdm(range(B), desc="Processing frames"):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], 
                    smpl_shape[f_id], 
                    smpl_transl[f_id], 
                    None, None, 
                    K=intrinsics_tosave[f_id].expand(n_humans_i, -1 , -1), 
                    expression=smpl_expression[f_id])
        
        if save or render:
            color = colors_tosave[f_id].numpy()
            c2w = cam2world_tosave[f_id].numpy()
            intrins = intrinsics_tosave[f_id].numpy()
        if save:
            depth = depths_tosave[f_id].numpy()
            conf = conf_self_tosave[f_id].numpy()

        if (save or save_json) and n_humans_i > 0:
            frame_key = str(f_id + frame_offset)
            camera_entry = {
                "x": float(c2w[0, 3] * axis_sign["x"]),
                "y": float(c2w[1, 3] * axis_sign["y"]),
                "z": float(c2w[2, 3] * axis_sign["z"]),
            }
            j3d_cam = smpl_out["smpl_j3d"].detach().cpu().numpy()
            j3d_world = (
                geotrf(pr_poses[f_id], smpl_out["smpl_j3d"].unsqueeze(0))[0]
                .detach()
                .cpu()
                .numpy()
            )
            names = joint_names[: j3d_cam.shape[1]]
            person_ids = (
                smpl_id[f_id].detach().cpu().numpy().tolist()
                if smpl_id[f_id].numel() > 0
                else list(range(n_humans_i))
            )
            for idx, pid in enumerate(person_ids[: j3d_cam.shape[0]]):
                human_key = get_human_entry(pid)
                if save_json:
                    joints_json_by_human[human_key]["frames"][frame_key] = {
                        "3d_joints": joints_to_dict(j3d_cam[idx], names),
                        "global_3d_joints": joints_to_dict(j3d_world[idx], names),
                    }
                else:
                    joints_json_by_human[human_key]["frames"][frame_key] = {
                        "camera": camera_entry.copy(),
                        "3d_joints": joints_to_dict(j3d_cam[idx], names),
                        "global_3d_joints": joints_to_dict(j3d_world[idx], names),
                    }

        if n_humans_i > 0:
            # transform smpl verts to world coordinates
            all_verts.append(geotrf(pr_poses[f_id], smpl_out['smpl_v3d'].unsqueeze(0))[0])
            pr_verts = [t.numpy() for t in smpl_out['smpl_v3d'].unbind(0)]
            pr_faces = [smpl_faces] * n_humans_i
        else:
            pr_verts = []
            pr_faces = []
            all_verts.append(torch.empty(0))

        if render:
            hm = vis_heatmap(colors_tosave[f_id], smpl_scores[f_id]).numpy()
            img_array_np = (color * 255).astype(np.uint8)
            smpl_rend = render_meshes(img_array_np.copy(), pr_verts, pr_faces,
                                        {'focal': intrins[[0,1],[0,1]], 
                                        'princpt': intrins[[0,1],[-1,-1]]},
                                        color=[get_color(i)/255 for i in smpl_id[f_id]])
            if has_mask:
                msk_array_np = vis_heatmap(colors_tosave[f_id], msks[f_id][0]).numpy()
                color_smpl = np.concatenate([
                    img_array_np, 
                    (msk_array_np * 255).astype(np.uint8), 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
            else:
                color_smpl = np.concatenate([
                    img_array_np, 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
        
        if save:
            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w,
                intrinsics=intrins,
            )
            np.savez(
                os.path.join(outdir, "smpl", f"{f_id:06d}.npz"),
                scores=smpl_scores[f_id].numpy(),
                msk=msks[f_id].numpy() if has_mask else None,
                shape=smpl_shape[f_id].numpy(),
                rotvec=smpl_rotvec[f_id].numpy(),
                transl=smpl_transl[f_id].numpy(),
                expression=smpl_expression[f_id].numpy() if smpl_expression[f_id] is not None else None
            )

        # Save smpl projection
        if render:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)
            iio.imwrite(
                os.path.join(outdir, "color_smpl", f"{f_id:06d}.png"),
                color_smpl,
            )

    print("prepare_output: 19")
    if save or save_json:
        json_dir = os.path.join(outdir, "json")
        json_suffix = f"_{json_frame_tag}" if json_frame_tag is not None else ""
        for human_key, data in joints_json_by_human.items():
            json_path = os.path.join(
                json_dir,
                f"joints_3d_human_{human_key}_{json_suffix}_original.json",
            )
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

    print("prepare_output: 20")
    if render and render_video:
        print(f"Saving smpl mesh projection to {outdir}...")
        frames_dir = os.path.join(outdir, "color_smpl")
        video_path = os.path.join(outdir, "output_video.mp4")
        output_fps = 30 // subsample
        os.system(f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"')
    
    print("prepare_output: 21")
    return (
        pts3ds_other,
        colors, 
        conf_other, 
        cam_dict, 
        all_verts, 
        smpl_faces,
        smpl_id,
        msks
    )

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in tqdm(frame_indices, desc="Extracting frames from video"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    try:
        args.model_path = _resolve_model_path(args)
    except Exception as err:
        print(f"モデルの準備に失敗しました: {err}")
        return

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    frame_offset = 0
    json_frame_tag = None
    if args.block_frame_num is not None:
        if args.block_frame_num <= 0:
            print("--block_frame_num は 1 以上の値を指定してください。")
            return
        if args.block_index < 0:
            print("--block_index は 0 以上の値を指定してください。")
            return
        total_frames = len(img_paths)
        block_start = args.block_index * args.block_frame_num
        block_end = block_start + args.block_frame_num
        if block_start >= total_frames:
            print(
                f"--block_index が範囲外です。block_start={block_start}, total_frames={total_frames}"
            )
            return
        img_paths = img_paths[block_start:block_end]
        frame_offset = block_start
        json_frame_tag = f"{block_start:05d}"
        block_end_actual = block_start + len(img_paths) - 1
        print(
            f"Block: index={args.block_index}, range={block_start}-{block_end_actual} (total={total_frames})"
        )

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    save_json = args.save_json
    save = args.save and not save_json
    render = args.render and not save_json
    render_video = args.render_video and not save_json
    if save_json:
        print("--save-json 指定のため、可視化と他の出力を無視して JSON のみ保存します。")

    # Process outputs for visualization / saving.
    print("Preparing output...")
    (
        pts3ds_other, 
        colors, 
        conf, 
        cam_dict, 
        all_smpl_verts, 
        smpl_faces,
        smpl_id,
        msks,
        ) = prepare_output(
        outputs,
        args.output_dir,
        1,
        True,
        save=save,
        render=render,
        render_video=render_video,
        img_res=img_res,
        subsample=args.subsample,
        save_json=save_json,
        frame_offset=frame_offset,
        json_frame_tag=json_frame_tag,
    )

    if not save and not save_json:
        from viser_utils import SceneHumanViewer
        # Convert tensors to numpy arrays for visualization.
        pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
        colors_to_vis = [c.cpu().numpy() for c in colors]
        msks_to_vis = [m.cpu().numpy() for m in msks]
        conf_to_vis = [c.cpu().numpy() for c in conf]
        edge_colors = [None] * len(pts3ds_to_vis)
        verts_to_vis = [p.cpu().numpy() for p in all_smpl_verts]

        # Create and run the point cloud viewer.
        print("Launching Human3R viewer...")
        viewer = SceneHumanViewer(
            pts3ds_to_vis,
            colors_to_vis,
            conf_to_vis,
            cam_dict,
            verts_to_vis,
            smpl_faces,
            smpl_id,
            msks_to_vis,
            device=device,
            edge_color_list=edge_colors,
            show_camera=True,
            vis_threshold=args.vis_threshold,
            msk_threshold=args.msk_threshold,
            mask_morph=args.mask_morph,
            size = args.size,
            downsample_factor=args.downsample_factor,
            smpl_downsample_factor=args.smpl_downsample,
            camera_downsample_factor=args.camera_downsample
        )
        viewer.run()

def convert_vmd(args):
    """
    json を vmd に変換します。
    """
    import subprocess

    out_path = os.path.join(args.output_dir, "json")
    subprocess.run(
        [
            f"{args.mat5_dir}/go/cmd/mat5",
            f"--modelPath={args.mat5_dir}/data/pmx/v4_trace_model.pmx",
            f"--dirPath={out_path}",
            "--logLevel=DEBUG",
        ]
    )


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        start_time = time.time()
        run_inference(args)
        convert_vmd(args)
        total_time = time.time() - start_time
        print(f"Human3R 処理終了: 合計時間 {total_time / 60:.2f} 分")


if __name__ == "__main__":
    main()

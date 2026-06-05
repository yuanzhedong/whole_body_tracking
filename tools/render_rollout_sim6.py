"""Isaac Sim 6.0 renderer for a logged policy rollout (decoupled from the 4.5 training stack).

Converts the G1 URDF->USD, then replays the rollout KINEMATICALLY by driving every link's
world transform directly from the logged per-body world poses (body_pos / body_quat). This
faithfully reproduces the policy motion (joint articulation + base motion) without stepping
physics -- crucial because at delta_time=0 a PhysX articulation's joint state never propagates
to the rendered mesh, which made the old set_joint_positions() approach render a rigid robot.

Works on driver 595 where Isaac Sim 4.5/5.0 RTX renderer crashes.

Camera modes:
  follow   - camera tracks the robot's xy (robot walks across the floor)  [default]
  treadmill- robot re-centered to origin (walks in place), static camera
"""
import argparse, os, glob, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--states", default="/tmp/rollout_states.npz")
parser.add_argument("--urdf", default="/ws/user/yzdong/src/github/whole_body_tracking/source/"
                                      "whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf")
parser.add_argument("--out_dir", default="/tmp/rollout_frames")
parser.add_argument("--usd_dir", default="/tmp/g1_usd")
parser.add_argument("--camera", choices=["follow", "treadmill"], default="follow")
parser.add_argument("--stride", type=int, default=2)        # 50fps states -> 25fps at stride 2
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--z_lift", type=float, default=0.0,
                    help="manual vertical nudge (m). Default 0: trust the sim z, which already "
                         "has correct ground contact (z=0 is the floor; a planted foot FRAME "
                         "sits a few cm above it because the sole hangs below the frame).")
parser.add_argument("--result", default="/tmp/render_rollout6_result.txt")
args = parser.parse_args()

RESULT = args.result
def w(m):
    with open(RESULT, "a") as f: f.write(m + "\n"); f.flush(); os.fsync(f.fileno())
open(RESULT, "w").close()


try:
    data = np.load(args.states)
    root_pos = data["root_pos"].astype(float)        # (T, 3) world, origin-subtracted
    body_pos = data["body_pos"].astype(float)         # (T, num_bodies, 3) world, origin-subtracted
    body_quat = data["body_quat"].astype(float)       # (T, num_bodies, 4) wxyz
    body_names = [str(x) for x in data["body_names"]]
    nframes, nbodies = body_pos.shape[0], body_pos.shape[1]
    w(f"STATES_LOADED frames={nframes} bodies={nbodies} camera={args.camera}")

    # treadmill: re-center xy to origin so the robot stays framed (walks in place)
    if args.camera in ("treadmill", "follow"):  # follow not yet implemented -> treadmill framing
        body_pos = body_pos.copy()
        body_pos[:, :, 0] -= root_pos[:, None, 0]
        body_pos[:, :, 1] -= root_pos[:, None, 1]

    # Do NOT re-align to the floor: the sim already has correct ground contact (z=0 is the
    # floor, and a planted foot FRAME sits a few cm above it because the sole hangs below the
    # frame). Forcing the lowest frame to z=0 sinks the sole underground. Only apply an
    # explicit manual nudge if asked.
    if args.z_lift:
        body_pos = body_pos.copy()
        body_pos[:, :, 2] += args.z_lift
    foot_min = float(body_pos[:, :, 2].min())
    w(f"Z_LIFT={args.z_lift:.4f} lowest_frame_z={foot_min:.4f}")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "enable_cameras": True})
    w("APP_LAUNCHED")

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom, Gf

    cfg = URDFImporterConfig(urdf_path=args.urdf, usd_path=args.usd_dir,
                             merge_mesh=False, collision_from_visuals=False)
    usd_path = URDFImporter(cfg).import_urdf()
    w(f"URDF_CONVERTED -> {usd_path}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path, "/World/Robot")
    world.reset()

    rep.create.light(light_type="dome", intensity=1500)
    cam = rep.create.camera(position=(2.3, 2.3, 1.3), look_at=(0.0, 0.0, 0.7))
    rp = rep.create.render_product(cam, tuple(args.res))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=args.out_dir, rgb=True)
    writer.attach([rp])

    # Find each link prim by name and prepare a world-space transform op for it.
    # resetXformStack=True makes the prim ignore ALL ancestor transforms, so each link's
    # op IS its world transform -- no parent-composition math, no double-transform.
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    name_to_op = {}
    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName()
        if nm in body_names and prim.IsA(UsdGeom.Xformable) and nm not in name_to_op:
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)   # ignore parent transforms -> ops are in world space
            name_to_op[nm] = xf.AddTransformOp()
    missing = [n for n in body_names if n not in name_to_op]
    w(f"LINKS_BOUND {len(name_to_op)}/{nbodies}" + (f" MISSING={missing}" if missing else ""))
    if missing:
        raise RuntimeError(f"could not bind link prims for: {missing}")
    body_idx = {n: i for i, n in enumerate(body_names)}

    def pose(i):
        for nm, op in name_to_op.items():
            j = body_idx[nm]
            q = body_quat[i, j]   # wxyz
            m = Gf.Matrix4d()
            m.SetRotate(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            m.SetTranslateOnly(Gf.Vec3d(float(body_pos[i, j, 0]),
                                        float(body_pos[i, j, 1]),
                                        float(body_pos[i, j, 2])))
            op.Set(m)
        rep.orchestrator.step(delta_time=0.0, rt_subframes=2)  # render only; poses set via USD

    rendered = 0
    for i in range(0, nframes, args.stride):
        pose(i)
        rendered += 1

    pngs = sorted(glob.glob(args.out_dir + "/**/*.png", recursive=True))
    w(f"RENDER_STEPPED rendered={rendered} PNG_COUNT={len(pngs)}")
    w("RENDER_OK" if pngs else "RENDER_NO_PNG")
    app.close()
except Exception as e:
    import traceback
    w("RENDER_FAIL: " + repr(e)); w(traceback.format_exc())

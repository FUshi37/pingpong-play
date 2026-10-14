"""Smoke-test batched MJX stepping for the ping-pong MuJoCo scene.

This is intentionally small: it checks whether the patched XML used by the
current SB3 environment can be copied to MJX, batched with vmap-style data, and
stepped under JIT. It is not a replacement for the Gym/SB3 environment yet.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


RL_SIM_DIR = Path(__file__).resolve().parent
DEFAULT_XML = RL_SIM_DIR / "moz1_pd.xml"
SIMULATION_BASE_MODES = (
    "legacy_floating",
    "aligned_floating",
    "aligned_fixed",
)
FIXED_BASE_JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "base_roll",
    "base_pitch",
    "base_yaw",
)
BALL_RACKET_COLLISION_GEOMETRIES = (
    "legacy_cylinder",
    "finite_mesh16",
)
BALL_RACKET_CONTACT_BOUNDARY_MODES = (
    "robot_dynamic",
    "fk_heavy_proxy_v1",
)
IDENTIFIED_RACKET_RADIUS_M = 0.075
IDENTIFIED_RACKET_HALF_THICKNESS_M = 0.001
RACKET_CONTACT_PROXY_BODY_NAME = "mjx_racket_contact_proxy"
RACKET_CONTACT_PROXY_JOINT_NAME = "mjx_racket_contact_proxy_free"
RACKET_CONTACT_PROXY_GEOM_NAME = "mjx_racket_contact_proxy_geom"
IDENTIFIED_RACKET_CONTACT_PROXY_MASS_KG = 1000.0


def _finite_racket_mesh16_vertices() -> str:
    """Return a deterministic 16-sided physical racket collision disk."""

    values = []
    for z_m in (
        -IDENTIFIED_RACKET_HALF_THICKNESS_M,
        IDENTIFIED_RACKET_HALF_THICKNESS_M,
    ):
        for index in range(16):
            angle = 2.0 * math.pi * index / 16
            values.extend((
                IDENTIFIED_RACKET_RADIUS_M * math.cos(angle),
                IDENTIFIED_RACKET_RADIUS_M * math.sin(angle),
                z_m,
            ))
    return " ".join(f"{value:.12g}" for value in values)


def _set_ball_racket_collision_geometry(
    root: ET.Element,
    geometry: str,
) -> None:
    """Apply an explicit opt-in collision geometry without touching legacy."""

    if geometry not in BALL_RACKET_COLLISION_GEOMETRIES:
        raise ValueError(
            "ball_racket_collision_geometry must be one of "
            f"{BALL_RACKET_COLLISION_GEOMETRIES}, got {geometry!r}"
        )
    if geometry == "legacy_cylinder":
        return
    rackets = [
        geom
        for geom in root.iter("geom")
        if geom.get("name") == "racket_rubber_fore"
    ]
    if len(rackets) != 1:
        raise ValueError("expected exactly one racket_rubber_fore geom")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    mesh_name = "racket_collision_mesh16"
    for old in list(asset.findall("mesh")):
        if old.get("name") == mesh_name:
            asset.remove(old)
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        vertex=_finite_racket_mesh16_vertices(),
    )
    racket = rackets[0]
    racket.set("type", "mesh")
    racket.set("mesh", mesh_name)
    racket.attrib.pop("size", None)


def _add_fk_heavy_racket_contact_proxy(
    root: ET.Element,
    *,
    mass_kg: float,
) -> None:
    """Add the V33-validated free-body contact boundary to a temporary XML."""

    if not math.isclose(
        float(mass_kg),
        IDENTIFIED_RACKET_CONTACT_PROXY_MASS_KG,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "fk_heavy_proxy_v1 requires the validated exact 1000 kg mass"
        )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("contact proxy requires a worldbody")
    if root.find(
        f".//body[@name='{RACKET_CONTACT_PROXY_BODY_NAME}']"
    ) is not None:
        raise ValueError("racket contact proxy already exists")
    mesh = root.find(".//mesh[@name='racket_collision_mesh16']")
    if mesh is None:
        raise ValueError("contact proxy requires finite mesh16 geometry")
    body = ET.SubElement(
        worldbody,
        "body",
        name=RACKET_CONTACT_PROXY_BODY_NAME,
        pos="0 0 0",
    )
    ET.SubElement(
        body,
        "freejoint",
        name=RACKET_CONTACT_PROXY_JOINT_NAME,
    )
    ET.SubElement(
        body,
        "geom",
        name=RACKET_CONTACT_PROXY_GEOM_NAME,
        type="mesh",
        mesh="racket_collision_mesh16",
        mass=f"{float(mass_kg):.12g}",
        rgba="0 0 0 0",
    )


def _require_mjx():
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except Exception as exc:  # pragma: no cover - import-time diagnostic
        raise SystemExit(
            "MJX/JAX is not available in this environment. Install it first, "
            "for example: pip install 'jax[cuda12]' mujoco-mjx\n"
            f"Import error: {exc}"
        ) from exc
    return jax, jnp, mujoco, mjx


def _batch_tree(jax, jnp, tree, n_envs: int):
    """Add a leading batch dimension to every JAX array leaf."""

    def batch_leaf(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jnp.broadcast_to(x, (n_envs,) + tuple(x.shape))
        return x

    return jax.tree_util.tree_map(batch_leaf, tree)


def _write_mjx_contact_only_xml(
    xml_path: Path,
    *,
    simulation_base_mode: str = "legacy_floating",
    ball_racket_collision_geometry: str = "legacy_cylinder",
    ball_racket_contact_boundary_mode: str = "robot_dynamic",
    ball_racket_contact_proxy_mass_kg: float = (
        IDENTIFIED_RACKET_CONTACT_PROXY_MASS_KG
    ),
) -> Path:
    """Write an MJX-compatible training collision XML.

    Visual mesh/cylinder/box geoms in the robot are kept for rendering but made
    non-colliding.  We then add a small set of sphere proxies for non-racket
    body contacts and explicit bitmasks for ball/racket/support contacts.  This
    preserves the task-relevant contact surface without asking MJX to compile
    unsupported broadphase pairs from the full MuJoCo scene.
    """

    if simulation_base_mode not in SIMULATION_BASE_MODES:
        raise ValueError(
            "simulation_base_mode must be one of "
            f"{SIMULATION_BASE_MODES}, got {simulation_base_mode!r}"
        )
    if (
        ball_racket_contact_boundary_mode
        not in BALL_RACKET_CONTACT_BOUNDARY_MODES
    ):
        raise ValueError(
            "ball_racket_contact_boundary_mode must be one of "
            f"{BALL_RACKET_CONTACT_BOUNDARY_MODES}"
        )
    if (
        ball_racket_contact_boundary_mode == "fk_heavy_proxy_v1"
        and ball_racket_collision_geometry != "finite_mesh16"
    ):
        raise ValueError("fk_heavy_proxy_v1 requires finite_mesh16")

    root = ET.parse(xml_path).getroot()
    _set_ball_racket_collision_geometry(
        root, ball_racket_collision_geometry
    )
    if ball_racket_contact_boundary_mode == "fk_heavy_proxy_v1":
        _add_fk_heavy_racket_contact_proxy(
            root,
            mass_kg=ball_racket_contact_proxy_mass_kg,
        )
    contact_racket_geom_name = (
        RACKET_CONTACT_PROXY_GEOM_NAME
        if ball_racket_contact_boundary_mode == "fk_heavy_proxy_v1"
        else "racket_rubber_fore"
    )

    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    for geom in root.iter("geom"):
        name = geom.get("name")
        if name == "ball":
            geom.set("contype", "1")
            geom.set("conaffinity", "6")
            geom.set("condim", "3")
        elif name == contact_racket_geom_name:
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("condim", "3")

    worldbody = root.find("worldbody")
    if worldbody is not None:
        floor = worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.set("contype", "8")
            floor.set("conaffinity", "8")
        for body in worldbody.iter("body"):
            if body.get("name") == "base":
                support = None
                for geom in body.findall("geom"):
                    if geom.get("name") == "mjx_base_support":
                        support = geom
                        break
                if support is None:
                    support = ET.SubElement(body, "geom", name="mjx_base_support")
                support.set("type", "box")
                support_z_m = (
                    -0.045
                    if simulation_base_mode == "legacy_floating"
                    else -0.055
                )
                support.set("pos", f"0 0 {support_z_m:.3f}")
                support.set("size", "0.28 0.28 0.045")
                support.set("rgba", "0 0 0 0")
                support.set("contype", "8")
                support.set("conaffinity", "8")
                support.set("friction", "0.8 0.1 0.1")
                support.set("condim", "3")
                break

        equality = root.find("equality")
        if equality is not None:
            for constraint in list(equality.findall("joint")):
                if constraint.get("name", "").startswith("mjx_fixed_base_"):
                    equality.remove(constraint)
        if simulation_base_mode == "aligned_fixed":
            if equality is None:
                equality = ET.SubElement(root, "equality")
            for joint_name in FIXED_BASE_JOINT_NAMES:
                ET.SubElement(
                    equality,
                    "joint",
                    name=f"mjx_fixed_base_{joint_name}",
                    joint1=joint_name,
                    polycoef="0 0 0 0 0",
                    solref="0.001 1",
                    solimp="0.9999 0.9999 0.001",
                )

        proxy_specs = {
            "waist03": [("torso", "0 0 0.02", "0.080")],
            "right03": [("upper_arm_a", "0 0 0", "0.055")],
            "right04": [("upper_arm_b", "0 0 0", "0.050")],
            "right05": [("forearm_a", "0 0 0", "0.045")],
            "right06": [("forearm_b", "0 0 0", "0.040")],
            "right07": [("wrist", "0 0 0", "0.035")],
            "left03": [("left_upper_arm_a", "0 0 0", "0.055")],
            "left04": [("left_upper_arm_b", "0 0 0", "0.050")],
            "head22": [("head", "0 -0.03 0", "0.055")],
        }
        for body in worldbody.iter("body"):
            body_name = body.get("name")
            if body_name not in proxy_specs:
                continue
            existing = {g.get("name") for g in body.findall("geom")}
            for label, pos, size in proxy_specs[body_name]:
                geom_name = f"mjx_ball_contact_{label}"
                if geom_name in existing:
                    proxy = body.find(f"./geom[@name='{geom_name}']")
                else:
                    proxy = ET.SubElement(body, "geom", name=geom_name)
                proxy.set("type", "sphere")
                proxy.set("pos", pos)
                proxy.set("size", size)
                proxy.set("rgba", "0 0 0 0")
                proxy.set("contype", "4")
                proxy.set("conaffinity", "1")
                proxy.set("friction", "0.2 0.001 0.0001")
                proxy.set("condim", "3")

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    for pair in list(contact.findall("pair")):
        g1, g2 = pair.get("geom1"), pair.get("geom2")
        if {g1, g2} != {"ball", contact_racket_geom_name}:
            contact.remove(pair)

    if not any(
        {p.get("geom1"), p.get("geom2")}
        == {"ball", contact_racket_geom_name}
        for p in contact.findall("pair")
    ):
        ET.SubElement(
            contact,
            "pair",
            geom1="ball",
            geom2=contact_racket_geom_name,
            friction="0.35 0.002 0.0001",
            solref="0.003 0.80",
            solimp="0.90 0.995 0.001",
            condim="3",
        )

    tmp = tempfile.NamedTemporaryFile(prefix="moz1_pingpong_mjx_", suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=False)
    return tmp_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal batched MJX/JAX step benchmark.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--full-collision",
        action="store_true",
        help="Use the unmodified patched XML collision masks. This may fail on MJX-JAX.",
    )
    args = parser.parse_args()

    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be positive")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    jax, jnp, mujoco, mjx = _require_mjx()

    # Reuse the same XML patching path as the current Gym environment so the
    # ball/racket/contact setup is tested, not just the raw robot XML.
    from rl_juggle_env_random import _build_temp_xml_with_ball

    patched_xml = _build_temp_xml_with_ball(args.xml.resolve())
    mjx_xml = patched_xml if args.full_collision else _write_mjx_contact_only_xml(patched_xml)
    mj_model = mujoco.MjModel.from_xml_path(str(mjx_xml))

    print(f"[mjx_smoke] JAX devices: {jax.devices()}")
    print(f"[mjx_smoke] XML: {args.xml}")
    print(f"[mjx_smoke] patched XML: {patched_xml}")
    if not args.full_collision:
        print(f"[mjx_smoke] MJX training collision XML: {mjx_xml}")
    print(f"[mjx_smoke] n_envs={args.n_envs}, steps={args.steps}")

    try:
        mjx_model = mjx.put_model(mj_model)
        base_data = mjx.make_data(mjx_model)
    except Exception as exc:
        raise SystemExit(
            "Failed to create MJX model/data. This usually means the MJCF uses "
            "a feature unsupported by MJX-JAX or the installed mujoco-mjx "
            f"version. Error: {exc}"
        ) from exc

    data = _batch_tree(jax, jnp, base_data, args.n_envs)

    batched_step = jax.vmap(lambda single_data: mjx.step(mjx_model, single_data))

    def rollout(d):
        def one_step(carry, _):
            return batched_step(carry), None

        d, _ = jax.lax.scan(one_step, d, None, length=args.steps)
        return d

    rollout_jit = jax.jit(rollout)

    print("[mjx_smoke] compiling/warming up...")
    for _ in range(max(1, args.warmup)):
        data = rollout_jit(data)
        jax.block_until_ready(data.qpos)

    t0 = time.perf_counter()
    data = rollout_jit(data)
    jax.block_until_ready(data.qpos)
    elapsed = time.perf_counter() - t0

    total_steps = args.n_envs * args.steps
    print(
        "[mjx_smoke] OK | "
        f"elapsed={elapsed:.4f}s | "
        f"aggregate_sps={total_steps / max(elapsed, 1e-9):,.0f}"
    )


if __name__ == "__main__":
    main()

"""Smoke-test batched MJX stepping for the ping-pong MuJoCo scene.

This is intentionally small: it checks whether the patched XML used by the
current SB3 environment can be copied to MJX, batched with vmap-style data, and
stepped under JIT. It is not a replacement for the Gym/SB3 environment yet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


RL_SIM_DIR = Path(__file__).resolve().parent
DEFAULT_XML = RL_SIM_DIR / "moz1_pd.xml"


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


def _write_mjx_contact_only_xml(xml_path: Path) -> Path:
    """Write an MJX-compatible training collision XML.

    Visual mesh/cylinder/box geoms in the robot are kept for rendering but made
    non-colliding.  We then add a small set of sphere proxies for non-racket
    body contacts and explicit bitmasks for ball/racket/support contacts.  This
    preserves the task-relevant contact surface without asking MJX to compile
    unsupported broadphase pairs from the full MuJoCo scene.
    """

    root = ET.parse(xml_path).getroot()

    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    for geom in root.iter("geom"):
        name = geom.get("name")
        if name == "ball":
            geom.set("contype", "1")
            geom.set("conaffinity", "6")
            geom.set("condim", "3")
        elif name == "racket_rubber_fore":
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
                support.set("pos", "0 0 -0.045")
                support.set("size", "0.28 0.28 0.045")
                support.set("rgba", "0 0 0 0")
                support.set("contype", "8")
                support.set("conaffinity", "8")
                support.set("friction", "0.8 0.1 0.1")
                support.set("condim", "3")
                break

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
        if {g1, g2} != {"ball", "racket_rubber_fore"}:
            contact.remove(pair)

    if not any({p.get("geom1"), p.get("geom2")} == {"ball", "racket_rubber_fore"} for p in contact.findall("pair")):
        ET.SubElement(
            contact,
            "pair",
            geom1="ball",
            geom2="racket_rubber_fore",
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

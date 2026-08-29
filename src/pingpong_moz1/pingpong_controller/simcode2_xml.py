"""Runtime copy of the simcode2 launch19 XML preprocessing contract."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SPORT_TASKSPACE_FIT_RIGHT_ARM_PD: dict[str, tuple[float, float]] = {
    "RightArm-0": (120000.0, 1250.0),
    "RightArm-1": (120000.0, 675.0),
    "RightArm-2": (101250.0, 1087.5),
    "RightArm-3": (75000.0, 843.75),
    "RightArm-4": (48750.0, 468.75),
    "RightArm-5": (56250.0, 531.25),
    "RightArm-6": (37500.0, 328.125),
}

RECOVERED_RMP_RMPMD_V2_RIGHT_ARM_PD: dict[str, tuple[float, float]] = {
    "RightArm-0": (112320.0, 9048.0),
    "RightArm-1": (184320.0, 9604.8),
    "RightArm-2": (133650.0, 9570.0),
    "RightArm-3": (73200.0, 4071.6),
    "RightArm-4": (30420.0, 3712.0),
    "RightArm-5": (62100.0, 5000.0),
    "RightArm-6": (35100.0, 2030.0),
}


def _apply_right_arm_pd_profile(
    root: ET.Element,
    profile: str,
) -> None:
    canonical = str(profile).strip().lower()
    if canonical in {"", "xml", "current"}:
        return
    profiles = {
        "sport_taskspace_fit_v1": SPORT_TASKSPACE_FIT_RIGHT_ARM_PD,
        "recovered_rmp_rmpmd_v2": RECOVERED_RMP_RMPMD_V2_RIGHT_ARM_PD,
    }
    if canonical not in profiles:
        raise ValueError(f"unknown right_arm_pd_profile={profile!r}")
    pd_values = profiles[canonical]
    patched = 0
    for actuator in root.findall(".//actuator/position"):
        name = actuator.get("name", "")
        if name not in pd_values:
            continue
        kp, kv = pd_values[name]
        actuator.set("kp", f"{kp:g}")
        actuator.set("kv", f"{kv:g}")
        patched += 1
    if patched != len(pd_values):
        raise ValueError(
            f"right_arm_pd_profile={profile!r} patched {patched} actuators, "
            f"expected {len(pd_values)}")


def build_simcode2_training_xml(
    xml_path: str | Path,
    *,
    right_arm_pd_profile: str = "xml",
) -> Path:
    """Return the exact CPU-loadable XML shape used by simcode2 MJX.

    This combines simcode2's ``_build_temp_xml_with_ball`` and
    ``_write_mjx_contact_only_xml`` transformations without importing JAX.
    Keeping it in the deployment package lets real FK and the CPU mirror use
    the same racket geometry, support contact, and warm-start posture as the
    checkpoint environment.
    """
    source = Path(xml_path).expanduser().resolve()
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir")
    if meshdir:
        meshdir_path = Path(meshdir).expanduser()
        compiler.set(
            "meshdir",
            str(meshdir_path if meshdir_path.is_absolute()
                else (source.parent / meshdir_path).resolve()),
        )
    else:
        compiler.set("meshdir", str((source.parent / "meshes").resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if not any(
            body.get("name") == "pingpong_ball"
            for body in worldbody.findall("body")):
        ball_body = ET.SubElement(
            worldbody, "body", name="pingpong_ball",
            pos="0.35 -0.12 1.05")
        ET.SubElement(ball_body, "freejoint", name="ball_free")
        ET.SubElement(
            ball_body,
            "geom",
            name="ball",
            type="sphere",
            size="0.02",
            mass="0.0027",
            rgba="1.0 0.55 0.1 1",
            contype="1",
            conaffinity="1",
            friction="0.20 0.001 0.0001",
            solref="0.003 0.80",
            solimp="0.90 0.995 0.001",
            margin="0.001",
            condim="3",
        )
    for body in worldbody.findall("body"):
        if body.get("name") != "pingpong_ball":
            continue
        for geom in body.findall("geom"):
            if geom.get("name") == "ball":
                geom.set("friction", "0.20 0.001 0.0001")
                geom.set("solref", "0.003 0.80")
                geom.set("solimp", "0.90 0.995 0.001")
                geom.set("margin", "0.001")
                geom.set("condim", "3")
                break
        break

    for body in list(worldbody.findall("body")):
        if body.get("name") == "table":
            worldbody.remove(body)
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "pingpong_table":
            worldbody.remove(geom)

    for body in worldbody.iter("body"):
        if body.get("name") != "right07":
            continue
        for geom in list(body.findall("geom")):
            if geom.get("name") == "right_ee_ball":
                body.remove(geom)
        racket = next((
            child for child in body
            if child.tag == "body" and child.get("name") == "right_racket"
        ), None)
        if racket is None:
            racket = ET.SubElement(
                body, "body", name="right_racket", pos="0 0 -0.215",
                quat="0.92388 0.38268 0 0")
            ET.SubElement(
                racket, "geom", name="racket_wood", type="cylinder",
                size="0.075 0.0055", pos="0 0 0",
                rgba="0.5 0.2 0.1 1")
            ET.SubElement(
                racket, "geom", name="racket_rubber_fore", type="cylinder",
                size="0.075 0.001", pos="0 0 0.0065",
                rgba="0.8 0.1 0.1 1")
        racket.set("pos", "0 0 -0.215")
        site = next((
            item for item in racket.findall("site")
            if item.get("name") == "right_ee_site"
        ), None)
        if site is None:
            site = ET.SubElement(
                racket, "site", name="right_ee_site", size="0.004",
                rgba="0 0 0 0")
        site.set("pos", "0 0 0.0065")
        for geom in racket.findall("geom"):
            if geom.get("name") == "racket_wood":
                geom.set("size", "0.075 0.0055")
                geom.set("pos", "0 0 0")
            elif geom.get("name") == "racket_rubber_fore":
                geom.set("size", "0.075 0.001")
                geom.set("pos", "0 0 0.0065")
                geom.set("friction", "0.35 0.002 0.0001")
                geom.set("solref", "0.003 0.80")
                geom.set("solimp", "0.90 0.995 0.001")
                geom.set("margin", "0.001")
                geom.set("condim", "3")
        break

    # MJX launch19 collision contract: visual robot geoms are non-colliding;
    # the ball sees only the racket and compact body proxies.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for geom in root.iter("geom"):
        if geom.get("name") == "ball":
            geom.set("contype", "1")
            geom.set("conaffinity", "6")
            geom.set("condim", "3")
        elif geom.get("name") == "racket_rubber_fore":
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("condim", "3")
    floor = worldbody.find("./geom[@name='floor']")
    if floor is not None:
        floor.set("contype", "8")
        floor.set("conaffinity", "8")
    for body in worldbody.iter("body"):
        if body.get("name") != "base":
            continue
        support = next((
            geom for geom in body.findall("geom")
            if geom.get("name") == "mjx_base_support"
        ), None)
        if support is None:
            support = ET.SubElement(body, "geom", name="mjx_base_support")
        support.attrib.update({
            "type": "box",
            "pos": "0 0 -0.045",
            "size": "0.28 0.28 0.045",
            "rgba": "0 0 0 0",
            "contype": "8",
            "conaffinity": "8",
            "friction": "0.8 0.1 0.1",
            "condim": "3",
        })
        break
    proxy_specs = {
        "waist03": (("torso", "0 0 0.02", "0.080"),),
        "right03": (("upper_arm_a", "0 0 0", "0.055"),),
        "right04": (("upper_arm_b", "0 0 0", "0.050"),),
        "right05": (("forearm_a", "0 0 0", "0.045"),),
        "right06": (("forearm_b", "0 0 0", "0.040"),),
        "right07": (("wrist", "0 0 0", "0.035"),),
        "left03": (("left_upper_arm_a", "0 0 0", "0.055"),),
        "left04": (("left_upper_arm_b", "0 0 0", "0.050"),),
        "head22": (("head", "0 -0.03 0", "0.055"),),
    }
    for body in worldbody.iter("body"):
        for label, position, size in proxy_specs.get(
                body.get("name", ""), ()):
            geom_name = f"mjx_ball_contact_{label}"
            proxy = next((
                geom for geom in body.findall("geom")
                if geom.get("name") == geom_name
            ), None)
            if proxy is None:
                proxy = ET.SubElement(body, "geom", name=geom_name)
            proxy.attrib.update({
                "type": "sphere",
                "pos": position,
                "size": size,
                "rgba": "0 0 0 0",
                "contype": "4",
                "conaffinity": "1",
                "friction": "0.2 0.001 0.0001",
                "condim": "3",
            })

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if {pair.get("geom1"), pair.get("geom2")} != {
                "ball", "racket_rubber_fore"}:
            contact.remove(pair)
    if not any(
            {pair.get("geom1"), pair.get("geom2")}
            == {"ball", "racket_rubber_fore"}
            for pair in contact.findall("pair")):
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

    # simcode5 applies this profile after adding the ball and before loading
    # the MuJoCo/MJX model.  Apply it to the same final XML tree for the CPU
    # mirror so both backends receive identical actuator kp/kv settings.
    _apply_right_arm_pd_profile(root, right_arm_pd_profile)

    handle = tempfile.NamedTemporaryFile(
        prefix="pingpong_simcode2_training_", suffix=".xml", delete=False)
    output = Path(handle.name)
    handle.close()
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=False)
    return output

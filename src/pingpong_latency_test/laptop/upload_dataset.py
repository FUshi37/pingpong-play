#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, field_validator


class LeRobotFeature(BaseModel):
    dtype: str
    shape: tuple[int, ...]
    names: list[str]
    info: dict[str, Any] | None = None

    @field_validator("names", mode="before")
    @classmethod
    def validate_names(cls, v: str | list[str] | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class LeRobotMeta(BaseModel):
    codebase_version: str
    robot_type: str
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: dict[str, LeRobotFeature]


class LeRobotDatasetInfo(LeRobotMeta):
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    splits: dict[str, str]


class CreateCollectionRequest(BaseModel):
    task_id: int
    assistant_id: int | None = None
    robot_type: str | None = None
    robot_name: str | None = None
    capture_device_sn: str | None = None
    device_ip: str | None = None
    movax_version: str | None = None
    param_set_version: str | None = None
    capturex_version: str | None = None
    mozbrain_version: str | None = None


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LeRobotDataCollector:
    # HTTP 请求超时设置（秒）
    REQUEST_TIMEOUT = 30

    def __init__(self, api_base_url: str) -> None:
        """初始化数据收集器"""
        self.api_base_url = api_base_url
        self.headers: dict[str, str] = {}
        self.token: str | None = None

    def login(self, phone: str, password: str) -> None:
        """登录获取 token"""
        # 构建登录 URL，需要从 data_collections API 回到根路径
        base_url = self.api_base_url.rsplit("/data_collections", 1)[0]
        url = f"{base_url}/auth/login"

        payload = {"password": password, "phone": phone}
        headers = {"Content-Type": "application/json"}

        logger.info(f"正在登录: {phone}")
        response = requests.post(url, json=payload, headers=headers, timeout=self.REQUEST_TIMEOUT)
        response.raise_for_status()

        resp_json = response.json()
        logger.debug(f"登录响应: {resp_json}")

        if resp_json.get("code", -1) != 0:
            error_msg = f"登录接口返回错误: {resp_json.get('msg', 'Unknown error')}"
            raise RuntimeError(error_msg)

        # 检查响应格式
        if "data" not in resp_json or "token" not in resp_json["data"]:
            error_msg = "登录响应格式错误: 缺少token字段"
            raise RuntimeError(error_msg)

        self.token = resp_json["data"]["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

        user_info = resp_json["data"].get("user", {})
        account = user_info.get("account", phone)
        expires_at = resp_json["data"].get("expiresAt")
        logger.info(f"登录成功，用户: {account}, Token过期时间: {expires_at}")

    def create_collection(self, task_id: int, request: CreateCollectionRequest) -> int:
        """创建采集批次"""
        url = f"{self.api_base_url}/collecting/collection"
        params = {"task_id": task_id}
        resp = requests.post(
            url,
            headers=self.headers,
            params=params,
            timeout=self.REQUEST_TIMEOUT,
            json=request.model_dump(mode="json"),
        )
        resp.raise_for_status()
        return int(resp.json()["data"]["collection_id"])

    def create_record(self, collection_id: int, record_data: dict) -> tuple[int, str]:
        """创建记录并获取上传URL"""
        url = f"{self.api_base_url}/collecting/collection/{collection_id}/record"
        resp = requests.post(url, headers=self.headers, json=record_data, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()["data"]
        return result["record_id"], result["upload_url"]

    def submit_collection_trace(self, collection_id: int, trace_data: dict) -> None:
        """提交采集过程操作记录"""
        url = f"{self.api_base_url}/collecting/collection/{collection_id}/trace"
        resp = requests.post(url, headers=self.headers, json=trace_data, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()

    def finish_collecting(self, collection_id: int, dataset_info: LeRobotDatasetInfo) -> None:
        """结束采集批次"""
        url = f"{self.api_base_url}/collecting/collection/{collection_id}/finish"
        resp = requests.post(
            url, headers=self.headers, json=dataset_info.model_dump(), timeout=self.REQUEST_TIMEOUT
        )
        resp.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot数据集上传工具")
    parser.add_argument("--dataset-dir", required=True, help="LeRobot数据集目录路径")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1/data_collections",
        help="API基础URL",
    )
    parser.add_argument("--phone", help="登录手机号（dry-run模式下可省略）")
    parser.add_argument("--password", default="Spirit-ai@2025", help="登录密码（默认：Spirit-ai@2025）")
    parser.add_argument("--task-id", type=int, help="任务ID（dry-run模式下可省略）")
    parser.add_argument("--collection-id", type=int, help="采集批次ID，如果不指定，则创建新的采集批次")
    parser.add_argument(
        "--episodes", type=int, nargs="*", help="要上传的Episode列表，如果不指定，则上传所有Episode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查数据集内容并生成报告，不实际上传",
    )
    # 降级方案的默认值参数
    parser.add_argument("--robot-type", default="Moz1", help="机器人类型（默认：Moz1）")
    parser.add_argument("--robot-name", default="unknown", help="机器人类型（默认：unknown）")
    parser.add_argument("--host-ip", default="0.0.0.0", help="采集主机IP（降级方案使用，默认：0.0.0.0）")
    parser.add_argument("--movax-version", default=None, help="Movax版本（降级方案使用，默认：None）")
    parser.add_argument(
        "--is-mistake",
        action="store_true",
        help="标记为错误数据（降级方案使用，默认：False）",
    )
    return parser.parse_args()


def load_dataset_info(dataset_dir: Path) -> LeRobotDatasetInfo | None:
    """从数据集中提取LeRobotDatasetInfo信息"""
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            return LeRobotDatasetInfo.model_validate(json.load(f))
    return None


def load_event_log(dataset_dir: Path) -> dict[str, list[dict]]:
    """加载event_log.jsonl文件并按类型分类"""
    event_log_path = dataset_dir / "event_log.jsonl"
    events: dict[str, list[dict]] = {"trace": [], "record": []}

    if not event_log_path.exists():
        logger.warning(f"事件日志文件不存在: {event_log_path}")
        return events

    with open(event_log_path, encoding="utf-8") as f:
        for raw_line in f:
            line_content = raw_line.strip()
            if not line_content:
                continue
            try:
                event = json.loads(line_content)
                endpoint = event.get("endpoint", "")
                payload = event.get("payload", {})

                if endpoint == "trace":
                    events["trace"].append(payload)
                elif endpoint == "record":
                    events["record"].append(payload)
            except json.JSONDecodeError as e:
                logger.warning(f"解析事件日志行失败: {e}, 行内容前50字符: {line_content[:50]}")
                continue

    logger.info(f"加载了 {len(events['trace'])} 条trace事件和 {len(events['record'])} 条record事件")
    return events


def load_full_file_content(file_path: Path) -> str:
    """加载整个文件内容作为字符串"""
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return ""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def load_specific_line(file_path: Path, line_idx: int) -> str:
    """加载文件中特定行的内容"""
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return ""

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_idx:
                return line.strip()

    logger.error(f"行索引 {line_idx} 超出文件范围: {file_path}")
    return ""


def find_episode_files(dataset_dir: Path, episode_idx: int) -> list[Path]:
    """查找与特定episode相关的所有parquet和mp4文件"""
    episode_files = []
    episode_pattern = f"episode_{episode_idx:06d}"

    # 查找parquet文件
    data_dir = dataset_dir / "data"
    if data_dir.exists():
        for chunk_dir in data_dir.glob("chunk-*"):
            parquet_file = chunk_dir / f"{episode_pattern}.parquet"
            if parquet_file.exists():
                episode_files.append(parquet_file)

    # 查找mp4文件
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        for chunk_dir in videos_dir.glob("chunk-*"):
            for camera_dir in chunk_dir.glob("*"):
                if camera_dir.is_dir():
                    mp4_file = camera_dir / f"{episode_pattern}.mp4"
                    if mp4_file.exists():
                        episode_files.append(mp4_file)

    return episode_files


def get_parquet_modification_time(dataset_dir: Path, episode_idx: int) -> datetime | None:
    """获取指定episode的parquet文件修改时间作为采集结束时间"""
    episode_pattern = f"episode_{episode_idx:06d}"
    data_dir = dataset_dir / "data"

    if not data_dir.exists():
        return None

    for chunk_dir in data_dir.glob("chunk-*"):
        parquet_file = chunk_dir / f"{episode_pattern}.parquet"
        if parquet_file.exists():
            # 获取文件修改时间戳并转换为datetime对象
            mtime = os.path.getmtime(parquet_file)
            return datetime.fromtimestamp(mtime)

    return None


def create_tar_package(files: list[Path], dataset_dir: Path) -> Path:
    """将文件打包成tar文件"""
    # 创建临时文件
    fd, temp_path = tempfile.mkstemp(suffix=".tar")
    os.close(fd)
    tar_path = Path(temp_path)

    with tarfile.open(str(tar_path), "w") as tar:
        for file_path in files:
            # 计算相对路径作为归档中的路径
            rel_path = file_path.relative_to(dataset_dir)
            tar.add(str(file_path), arcname=str(rel_path))

    return tar_path


def upload_tar_to_url(tar_path: Path, upload_url: str) -> bool:
    """将tar文件上传到指定URL"""
    try:
        file_size_mb = tar_path.stat().st_size / (1024 * 1024)
        # 根据文件大小动态设置超时：基础30秒 + 每MB额外5秒，最多10分钟
        timeout = min(30 + int(file_size_mb * 5), 600)
        logger.debug(f"上传文件大小: {file_size_mb:.2f}MB, 超时设置: {timeout}秒")

        with open(str(tar_path), "rb") as f:
            resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "application/x-tar"},
                timeout=timeout,
            )
            resp.raise_for_status()
    except Exception:
        logger.exception("上传失败")
        return False
    else:
        return True
    finally:
        # 删除临时tar文件
        tar_path.unlink(missing_ok=True)


def check_dataset(dataset_dir: Path) -> dict[str, Any]:
    """检查数据集内容并返回报告"""
    report: dict[str, Any] = {
        "has_event_log": False,
        "total_episodes": 0,
        "episodes_with_event_log": 0,
        "episodes_without_event_log": 0,
        "total_trace_events": 0,
        "missing_files": [],
        "episodes_detail": [],
    }

    # 检查 event_log
    event_log_path = dataset_dir / "event_log.jsonl"
    if event_log_path.exists():
        report["has_event_log"] = True
        events = load_event_log(dataset_dir)
        record_events = {event["episode_idx"]: event for event in events["record"]}
        report["total_trace_events"] = len(events["trace"])
    else:
        record_events = {}

    # 加载数据集信息
    dataset_info = load_dataset_info(dataset_dir)
    if dataset_info is None:
        return report

    report["total_episodes"] = dataset_info.total_episodes

    # 检查每个 episode
    for episode_idx in range(dataset_info.total_episodes):
        episode_detail: dict[str, Any] = {
            "episode_idx": episode_idx,
            "has_event_log_record": episode_idx in record_events,
            "has_parquet": False,
            "parquet_mtime": None,
            "has_video": False,
            "video_count": 0,
        }

        # 检查 parquet 文件
        parquet_mtime = get_parquet_modification_time(dataset_dir, episode_idx)
        if parquet_mtime:
            episode_detail["has_parquet"] = True
            episode_detail["parquet_mtime"] = parquet_mtime.isoformat()

        # 检查相关文件
        episode_files = find_episode_files(dataset_dir, episode_idx)
        video_files = [f for f in episode_files if f.suffix == ".mp4"]
        episode_detail["has_video"] = len(video_files) > 0
        episode_detail["video_count"] = len(video_files)

        if episode_detail["has_event_log_record"]:
            report["episodes_with_event_log"] += 1
        else:
            report["episodes_without_event_log"] += 1

        if not episode_files:
            report["missing_files"].append(episode_idx)

        report["episodes_detail"].append(episode_detail)

    return report


def print_dataset_report(report: dict[str, Any]) -> None:
    """打印数据集检查报告"""
    logger.info("\n数据集检查报告:")
    logger.info(f"总 Episode 数: {report['total_episodes']}")
    logger.info(f"Event Log 存在: {'是' if report['has_event_log'] else '否'}")

    if report["has_event_log"]:
        logger.info(f"  - 有 event_log 记录的 Episode: {report['episodes_with_event_log']}")
        logger.info(f"  - 无 event_log 记录的 Episode: {report['episodes_without_event_log']}")
        logger.info(f"  - 总 Trace 事件数: {report['total_trace_events']}")

    if report["missing_files"]:
        logger.warning(f"\n缺少文件的 Episode ({len(report['missing_files'])} 个):")
        for episode_idx in report["missing_files"]:
            logger.warning(f"  - Episode {episode_idx}")

    # 打印无 event_log 记录的 episode 详情
    episodes_without_log = [ep for ep in report["episodes_detail"] if not ep["has_event_log_record"]]
    if episodes_without_log:
        logger.info(f"\n无 event_log 记录的 Episode 详情 ({len(episodes_without_log)} 个):")
        for ep in episodes_without_log:
            logger.info(f"  Episode {ep['episode_idx']}:")
            logger.info(f"    - 有 parquet 文件: {'是' if ep['has_parquet'] else '否'}")
            if ep["has_parquet"]:
                logger.info(f"    - Parquet 修改时间: {ep['parquet_mtime']}")
            logger.info(f"    - 视频文件数量: {ep['video_count']}")


def print_failure_report(failures: dict[str, list[dict]]) -> None:
    """打印失败报告"""
    total_failures = sum(len(v) for v in failures.values())
    if total_failures == 0:
        logger.info("所有操作均成功完成")
        return

    logger.error(f"\n执行过程中发生 {total_failures} 个失败\n")

    if failures["record_creation"]:
        logger.error(f"Record 创建失败 ({len(failures['record_creation'])} 个):")
        for failure in failures["record_creation"]:
            logger.error(f"  - Episode {failure['episode_idx']}: {failure['error']}")

    if failures["trace_submission"]:
        logger.error(f"\nTrace 提交失败 ({len(failures['trace_submission'])} 个):")
        for failure in failures["trace_submission"]:
            logger.error(
                f"  - Episode {failure['episode_idx']}, "
                f"Trace类型 {failure['trace_type']}: {failure['error']}"
            )

    if failures["file_upload"]:
        logger.error(f"\n文件上传失败 ({len(failures['file_upload'])} 个):")
        for failure in failures["file_upload"]:
            logger.error(f"  - Episode {failure['episode_idx']}: {failure['error']}")

    if failures["collection_finish"]:
        logger.error(f"\nCollection 完成失败 ({len(failures['collection_finish'])} 个):")
        for failure in failures["collection_finish"]:
            logger.error(f"  - Collection {failure['collection_id']}: {failure['error']}")


def prepare_record_data_with_event_log(
    episode_idx: int,
    record_events: dict[int, dict],
    collection_id: int,
) -> dict:
    """使用 event_log 中的数据准备 record 数据"""
    record_data = record_events[episode_idx].copy()
    record_data["collection_id"] = collection_id
    logger.info(f"使用event_log中的完整数据创建Episode {episode_idx}")
    return record_data


def prepare_record_data_fallback(
    episode_idx: int,
    dataset_dir: Path,
    task_id: int,
    collection_id: int,
    hostname: str = "unknown",
    host_ip: str = "0.0.0.0",
    is_mistake: bool = False,
) -> dict:
    """降级方案：使用parquet文件时间和默认值准备 record 数据"""
    logger.warning(f"event_log中未找到Episode {episode_idx} 的数据，使用降级方案")
    meta_dir = dataset_dir / "meta"
    info_json = load_full_file_content(meta_dir / "info.json")
    tasks_jsonl = load_full_file_content(meta_dir / "tasks.jsonl")
    episodes_jsonl = load_specific_line(meta_dir / "episodes.jsonl", episode_idx)
    episodes_stats_jsonl = load_specific_line(meta_dir / "episodes_stats.jsonl", episode_idx)

    # 获取parquet文件修改时间作为采集结束时间
    ended_at = get_parquet_modification_time(dataset_dir, episode_idx)
    if ended_at:
        logger.info(f"Episode {episode_idx} - 使用parquet文件时间作为ended_at: {ended_at}")
    else:
        logger.warning(f"Episode {episode_idx} - 未找到parquet文件，ended_at将为None")

    return {
        "task_id": task_id,
        "collection_id": collection_id,
        "episode_idx": episode_idx,
        "tasks_jsonl": tasks_jsonl,
        "info_json": info_json,
        "episodes_jsonl": episodes_jsonl,
        "episodes_stats_jsonl": episodes_stats_jsonl,
        "hostname": hostname,
        "host_ip": host_ip,
        "started_at": None,
        "ended_at": ended_at.isoformat() if ended_at else None,
        "is_mistake": is_mistake,
    }


def process_single_episode(
    episode_idx: int,
    collection_id: int,
    collector: LeRobotDataCollector,
    dataset_dir: Path,
    record_events: dict[int, dict],
    trace_events_by_recording: dict[int, list[dict]],
    task_id: int,
    failures: dict[str, list[dict]],
    hostname: str = "unknown",
    host_ip: str = "0.0.0.0",
    is_mistake: bool = False,
    only_with_event_log: bool = False,
) -> None:
    """处理单个 episode：创建 record、上传 trace、上传文件"""
    # 准备 record 数据
    has_event_log_record = episode_idx in record_events
    if has_event_log_record:
        record_data = prepare_record_data_with_event_log(episode_idx, record_events, collection_id)
    elif not only_with_event_log:
        record_data = prepare_record_data_fallback(
            episode_idx, dataset_dir, task_id, collection_id, hostname, host_ip, is_mistake
        )
    else:
        logger.error(f"Episode {episode_idx} - 无event_log数据，且only_with_event_log为True，跳过处理")
        return

    # 创建记录
    try:
        record_id, upload_url = collector.create_record(collection_id, record_data)
        logger.info(f"Episode {episode_idx} - 创建记录ID: {record_id}")
    except Exception as e:
        logger.exception(f"创建Episode {episode_idx} 的记录失败")
        failures["record_creation"].append({"episode_idx": episode_idx, "error": str(e)})
        return

    # 上传 trace 事件
    if has_event_log_record and episode_idx in trace_events_by_recording:
        trace_count = 0
        for trace_event in trace_events_by_recording[episode_idx]:
            try:
                collector.submit_collection_trace(collection_id, trace_event)
                trace_count += 1
                logger.debug(
                    f"已提交trace事件: {trace_event.get('type')}, "
                    f"recording_idx={trace_event.get('recording_idx')}"
                )
            except Exception as e:
                logger.error(
                    f"提交trace事件失败 - Episode {episode_idx}, "
                    f"类型: {trace_event.get('type', 'unknown')}, 错误: {e}"
                )
                failures["trace_submission"].append(
                    {
                        "episode_idx": episode_idx,
                        "trace_type": trace_event.get("type", "unknown"),
                        "error": str(e),
                    }
                )
        logger.info(f"Episode {episode_idx} - 成功提交 {trace_count} 条trace事件")
    elif not has_event_log_record:
        logger.info(f"Episode {episode_idx} - 无event_log数据，跳过trace事件上传")

    # 查找并上传文件
    episode_files = find_episode_files(dataset_dir, episode_idx)
    if not episode_files:
        logger.warning(f"未找到Episode {episode_idx} 的相关文件")
        return

    logger.info(f"找到Episode {episode_idx} 的文件数量: {len(episode_files)}")
    tar_path = create_tar_package(episode_files, dataset_dir)
    logger.info(f"已创建tar包: {tar_path}")

    success = upload_tar_to_url(tar_path, upload_url)
    if success:
        logger.info(f"成功上传Episode {episode_idx} 的数据")
    else:
        logger.error(f"上传Episode {episode_idx} 的数据失败")
        failures["file_upload"].append(
            {
                "episode_idx": episode_idx,
                "error": "文件上传失败（详见上方日志）",
            }
        )


def upload_all_episodes(
    dataset_info: LeRobotDatasetInfo,
    collection_id: int,
    collector: LeRobotDataCollector,
    dataset_dir: Path,
    record_events: dict[int, dict],
    trace_events_by_recording: dict[int, list[dict]],
    task_id: int,
    failures: dict[str, list[dict]],
    hostname: str = "unknown",
    host_ip: str = "0.0.0.0",
    is_mistake: bool = False,
    only_with_event_log: bool = False,
    episodes: list[int] | None = None,
) -> None:
    """上传所有 episodes"""
    total_episodes = dataset_info.total_episodes
    logger.info(f"开始上传 {total_episodes} 个 episodes")

    if episodes is None:
        episodes = list(range(total_episodes))
    for i, episode_idx in enumerate(episodes):
        logger.info(f"上传进度: {i + 1}/{len(episodes)}")
        process_single_episode(
            episode_idx,
            collection_id,
            collector,
            dataset_dir,
            record_events,
            trace_events_by_recording,
            task_id,
            failures,
            hostname,
            host_ip,
            is_mistake,
            only_with_event_log,
        )

    logger.info(f"完成上传所有 episodes，成功 {total_episodes - len(failures['record_creation'])} 个")


def main() -> int:
    """主函数，返回退出码"""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    # Dry run 模式：只检查数据集并生成报告
    if args.dry_run:
        logger.info("Dry run 模式：只检查数据集内容，不实际上传")
        report = check_dataset(dataset_dir)
        if report["total_episodes"] == 0:
            logger.error(f"数据集信息文件不存在: {dataset_dir / 'meta' / 'info.json'}")
            return 2
        print_dataset_report(report)
        return 1 if report["missing_files"] else 0

    # 验证必需参数
    if not args.phone or not args.task_id:
        logger.error("非 dry-run 模式下必须提供 --phone 和 --task-id")
        return 2

    # 加载数据集信息
    dataset_info = load_dataset_info(dataset_dir)
    if dataset_info is None:
        logger.error(f"数据集信息文件不存在: {dataset_dir / 'meta' / 'info.json'}")
        return 2
    logger.info(f"Episode 数量: {dataset_info.total_episodes}")

    # 加载事件日志并分组
    events = load_event_log(dataset_dir)
    record_events = {event["episode_idx"]: event for event in events["record"]}
    trace_events_by_recording: dict[int, list[dict]] = {}
    for trace_event in events["trace"]:
        recording_idx = trace_event.get("recording_idx")
        if recording_idx is not None:
            trace_events_by_recording.setdefault(recording_idx, []).append(trace_event)

    # 验证 event log 中的 record 数量和 episode 数量是否一致
    if len(events) > 0:
        record_count = len(record_events)
        episode_count = dataset_info.total_episodes
        if record_count != episode_count:
            logger.error(
                f"event_log 中的 record 数量 ({record_count}) "
                f"与数据集的 episode 数量 ({episode_count}) 不一致，请检查数据集内容：{dataset_dir}"
            )

    # 初始化数据收集器并登录
    collector = LeRobotDataCollector(args.api_url)
    try:
        collector.login(args.phone, args.password)
    except Exception:
        logger.exception("登录失败")
        return 2

    if args.collection_id:
        collection_id = args.collection_id
    else:
        # 创建采集批次
        create_collection_request = CreateCollectionRequest(
            task_id=args.task_id,
            robot_type=args.robot_type.lower(),
            robot_name=args.robot_name,
            capture_device_sn=args.robot_name.lower(),
            device_ip=args.host_ip,
            movax_version=args.movax_version,
        )
        try:
            collection_id = collector.create_collection(args.task_id, create_collection_request)
            logger.info(f"创建采集批次ID: {collection_id}")
        except Exception:
            logger.exception("创建采集批次失败，请求参数: %s", create_collection_request.model_dump_json())
            return 2

    # 上传所有 episodes
    failures: dict[str, list[dict]] = defaultdict(list)
    upload_all_episodes(
        dataset_info,
        collection_id,
        collector,
        dataset_dir,
        record_events,
        trace_events_by_recording,
        args.task_id,
        failures,
        args.robot_name,
        args.host_ip,
        args.is_mistake,
        len(events) > 0,
        args.episodes,
    )

    # 结束采集批次
    try:
        collector.finish_collecting(collection_id, dataset_info)
        logger.info(f"成功完成采集批次 {collection_id}")
    except Exception as e:
        logger.exception("完成采集批次失败")
        failures["collection_finish"].append({"collection_id": collection_id, "error": str(e)})

    # 打印失败报告并返回退出码
    print_failure_report(failures)
    total_failures = sum(len(v) for v in failures.values())
    if total_failures == 0:
        return 0
    if failures["record_creation"] or failures["collection_finish"]:
        return 2
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

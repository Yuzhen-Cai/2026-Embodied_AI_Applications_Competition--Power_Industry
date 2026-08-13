#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自训练权重同步工具（阿里云 OSS · S3 兼容接口）。

约定：
    本地权重目录：outputs/<task>/<version>/
    云端对象前缀：weights/<task>/<version>/
    建议每个版本目录下附带 README.md，记录训练配置、指标与日期。

用法：
    python scripts/sync_weights.py push <task> <version>   # 上传本地版本到 OSS
    python scripts/sync_weights.py pull <task> <version>   # 从 OSS 下载版本到本地
    python scripts/sync_weights.py list [task]             # 列出 OSS 上已有的版本/任务

依赖：
    pip install boto3 python-dotenv

配置：
    复制项目根目录的 .env.example 为 .env，填入真实凭据（.env 已被 gitignore 忽略）。
    详见 docs/weight-sync-guide.md。
"""

import argparse
import os
import sys

import boto3
from botocore.client import Config
from dotenv import load_dotenv


def load_config():
    """从项目根目录的 .env 加载凭据，缺失时给出提示。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    load_dotenv(env_path)

    required = ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_BUCKET", "OSS_ENDPOINT"]
    cfg = {k: os.getenv(k) for k in required}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        print("[错误] 缺少以下配置项：", ", ".join(missing))
        print("请复制项目根目录的 .env.example 为 .env，并填入真实凭据。")
        sys.exit(1)

    cfg["OSS_REGION"] = os.getenv("OSS_REGION", "cn-hangzhou")
    return cfg


def make_client(cfg):
    """创建指向 OSS S3 兼容接口的 boto3 客户端。"""
    return boto3.client(
        "s3",
        endpoint_url="https://{}".format(cfg["OSS_ENDPOINT"]),
        aws_access_key_id=cfg["OSS_ACCESS_KEY_ID"],
        aws_secret_access_key=cfg["OSS_ACCESS_KEY_SECRET"],
        region_name=cfg["OSS_REGION"],
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_dir(client, bucket, local_dir, prefix):
    """递归上传 local_dir 到 bucket 下的 prefix。"""
    if not os.path.isdir(local_dir):
        print("[错误] 本地目录不存在：{}".format(local_dir))
        sys.exit(1)

    n = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = "{}/{}".format(prefix, rel)
            print("  上传 {}".format(rel))
            client.upload_file(local_path, bucket, key)
            n += 1
    return n


def download_dir(client, bucket, prefix, local_dir):
    """递归下载 bucket 中 prefix 下的对象到 local_dir。"""
    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, prefix).replace("\\", "/")
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print("  下载 {}".format(rel))
            client.download_file(bucket, key, local_path)
            n += 1
    return n


def list_versions(client, bucket, prefix):
    """按“目录”列出 prefix 下的直接子项（任务名或版本名）。"""
    paginator = client.get_paginator("list_objects_v2")
    result = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            result.append(cp["Prefix"].rstrip("/").split("/")[-1])
    return result


def main():
    parser = argparse.ArgumentParser(description="自训练权重同步工具（阿里云 OSS）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="上传本地版本到 OSS")
    p_push.add_argument("task", help="任务名，如 gait / vla")
    p_push.add_argument("version", help="版本名，如 ppo_v1_2026-08-13")

    p_pull = sub.add_parser("pull", help="从 OSS 下载版本到本地")
    p_pull.add_argument("task", help="任务名，如 gait / vla")
    p_pull.add_argument("version", help="版本名，如 ppo_v1_2026-08-13")

    p_list = sub.add_parser("list", help="列出 OSS 上已有的版本（不带 task 时列任务）")
    p_list.add_argument("task", nargs="?", default=None, help="可选，指定任务名")

    args = parser.parse_args()
    cfg = load_config()
    client = make_client(cfg)
    bucket = cfg["OSS_BUCKET"]

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    local_root = os.path.join(root, "outputs")
    cloud_root = "weights"

    if args.cmd == "push":
        local_dir = os.path.join(local_root, args.task, args.version)
        prefix = "{}/{}/{}".format(cloud_root, args.task, args.version)
        if not os.path.isfile(os.path.join(local_dir, "README.md")):
            print("[提示] 该版本目录下没有 README.md，建议补充训练配置与指标说明（见 docs/weight-sync-guide.md）。")
        print("[push] {} -> oss://{}/{}".format(local_dir, bucket, prefix))
        n = upload_dir(client, bucket, local_dir, prefix)
        print("完成，共上传 {} 个文件。".format(n))

    elif args.cmd == "pull":
        local_dir = os.path.join(local_root, args.task, args.version)
        prefix = "{}/{}/{}".format(cloud_root, args.task, args.version)
        print("[pull] oss://{}/{} -> {}".format(bucket, prefix, local_dir))
        n = download_dir(client, bucket, prefix, local_dir)
        if n == 0:
            print("[警告] 云端没有找到该版本，请先用 `list {}` 确认版本名。".format(args.task))
        else:
            print("完成，共下载 {} 个文件。".format(n))

    elif args.cmd == "list":
        prefix = "{}/{}/".format(cloud_root, args.task) if args.task else "{}/".format(cloud_root)
        items = list_versions(client, bucket, prefix)
        if args.task:
            print("任务 [{}] 的可用版本：".format(args.task))
        else:
            print("云端已有的任务：")
        for v in items:
            print("  - {}".format(v))


if __name__ == "__main__":
    main()

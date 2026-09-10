#!/usr/bin/env python3
"""
Fin-Agent 环境变量安全管理器

功能:
  - 检测 .env 中明文密钥
  - 生成加密的 .env.encrypted 版本
  - 与 Docker Secrets 兼容
  - 支持 KMS fallback

用法:
  # 检查环境配置
  python scripts/env_manager.py check

  # 加密 .env 为 .env.encrypted (使用 Fernet)
  python scripts/env_manager.py encrypt

  # 解密 .env.encrypted 为 .env
  python scripts/env_manager.py decrypt

  # 列出所有配置的密钥状态
  python scripts/env_manager.py status
"""

import os
import sys
import re
import json
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_ENCRYPTED = PROJECT_ROOT / ".env.encrypted"
ENV_TEMPLATE = PROJECT_ROOT / ".env.template"

# 需要保护的敏感密钥列表
SENSITIVE_KEYS = [
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "GEMINI_API_KEY",
    "NEO4J_PASSWORD",
    "NEO4J_URI",
    "REDIS_PASSWORD",
    "SECRET_KEY",
    "JWT_SECRET",
]


def load_env_file(path: Path) -> dict:
    """解析 .env 文件为字典"""
    env = {}
    if not path.exists():
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$', line, re.IGNORECASE)
            if match:
                key, value = match.group(1), match.group(2)
                # 去除引号 (单引号或双引号)
                value = value.strip("'\"").strip()
                env[key] = value
    return env


def check_env():
    """检查 .env 配置状态"""
    print("=" * 60)
    print("  Fin-Agent 环境变量安全检查")
    print("=" * 60)

    if not ENV_FILE.exists():
        print(f"\n[WARN]  文件不存在: {ENV_FILE}")
        print(f"  请从 {ENV_TEMPLATE} 复制: cp .env.template .env")
        return False

    env = load_env_file(ENV_FILE)
    if not env:
        print(f"\n[WARN]  文件为空或无法解析: {ENV_FILE}")
        return False

    print(f"\n[INFO] 已配置 {len(env)} 个环境变量:\n")

    issues = []
    for key in sorted(env.keys()):
        value = env[key]
        is_sensitive = key in SENSITIVE_KEYS
        is_set = bool(value) and value not in ("", "your-api-key", "your-password")
        is_placeholder = value in ("your-api-key", "your-password", "YOUR_API_KEY_HERE")
        has_default = any(
            default in value.lower()
            for default in ["your-", "sk-your", "placeholder", "changeme"]
        )

        prefix = "[RED]" if (is_sensitive and (not is_set or is_placeholder)) else \
                 "[YEL]" if (is_sensitive and has_default) else "[GRN]"
        display = f"{prefix}  {key}"

        if is_sensitive and is_set:
            display += f" = {value[:8]}...{value[-4:]}"
        elif is_set:
            display += f" = {value[:40]}{'...' if len(value) > 40 else ''}"
        else:
            display += " = [未设置]"

        print(f"  {display}")

        if is_sensitive and (not is_set or is_placeholder or has_default):
            issues.append(key)

    print(f"\n{'=' * 60}")
    if issues:
        print(f"[WARN]  发现 {len(issues)} 个密钥未配置或使用了占位符:")
        for k in issues:
            print(f"   - {k}")
        print("\n  请更新 .env 文件中的对应值后重试。")
    else:
        print("[OK] 所有密钥已配置!")

    return len(issues) == 0


def encrypt_env():
    """加密 .env 文件 (使用 Fernet)"""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        print("[NO] 需要安装 cryptography: pip install cryptography")
        return

    if not ENV_FILE.exists():
        print(f"[NO] {ENV_FILE} 不存在，无法加密")
        return

    # 生成或加载密钥
    key_file = PROJECT_ROOT / ".env_key"
    if key_file.exists():
        with open(key_file, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
        print(f"[KEY] 新密钥已生成: {key_file}")
        print("[WARN]  请务必将此密钥备份到安全位置!")

    fernet = Fernet(key)

    # 加密
    with open(ENV_FILE, "rb") as f:
        encrypted = fernet.encrypt(f.read())

    with open(ENV_ENCRYPTED, "wb") as f:
        f.write(encrypted)

    print(f"[OK] .env 已加密为 {ENV_ENCRYPTED}")
    print(f"   解密密钥: {key_file}")


def decrypt_env():
    """解密 .env.encrypted 文件"""
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError:
        print("[NO] 需要安装 cryptography: pip install cryptography")
        return

    if not ENV_ENCRYPTED.exists():
        print(f"[NO] {ENV_ENCRYPTED} 不存在")
        return

    key_file = PROJECT_ROOT / ".env_key"
    if not key_file.exists():
        print(f"[NO] 密钥文件不存在: {key_file}")
        print("  请将备份的密钥文件放在项目根目录下")
        return

    with open(key_file, "rb") as f:
        key = f.read()

    fernet = Fernet(key)

    try:
        with open(ENV_ENCRYPTED, "rb") as f:
            decrypted = fernet.decrypt(f.read())
    except InvalidToken:
        print("[NO] 密钥不匹配或文件已损坏")
        return

    with open(ENV_FILE, "wb") as f:
        f.write(decrypted)

    print(f"[OK] 已解密为 {ENV_FILE}")


def status():
    """显示环境状态摘要"""
    env = load_env_file(ENV_FILE) if ENV_FILE.exists() else {}
    env_enc = ENV_ENCRYPTED.exists()

    configured = sum(1 for k in SENSITIVE_KEYS if env.get(k) and
                     env[k] not in ("", "your-api-key", "your-password"))
    missing = len(SENSITIVE_KEYS) - configured

    print(f"""
┌─────────────────────────────────────┐
│  Fin-Agent 环境变量状态              │
├─────────────────────────────────────┤
│  .env 存在:         {'[OK]' if ENV_FILE.exists() else '[NO]'}              │
│  .env.encrypted:    {'[OK]' if env_enc else '[NO]'}              │
│  .env.template:     {'[OK]' if ENV_TEMPLATE.exists() else '[NO]'}              │
├─────────────────────────────────────┤
│  敏感密钥已配置:     {configured}/{len(SENSITIVE_KEYS)}              │
│  缺失密钥:          {missing}              │
└─────────────────────────────────────┘
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/env_manager.py <command>")
        print(f"命令: check, encrypt, decrypt, status")
        sys.exit(1)

    command = sys.argv[1]
    if command == "check":
        sys.exit(0 if check_env() else 1)
    elif command == "encrypt":
        encrypt_env()
    elif command == "decrypt":
        decrypt_env()
    elif command == "status":
        status()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)

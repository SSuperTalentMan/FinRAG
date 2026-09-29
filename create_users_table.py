#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_users_table.py — 创建 MySQL users 表（认证模块必需）
包含 role、email、avatar、status 字段，并初始化默认管理员账号。

配置统一从 config.py（config.ini + 环境变量）读取，避免把数据库地址/账号写死。
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path（支持从任意目录运行）
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pymysql
import bcrypt
from config import get_config

cfg = get_config()
MYSQL_CONFIG = {
    "host": cfg.mysql.host,
    "port": cfg.mysql.port,
    "user": cfg.mysql.user,
    "password": cfg.mysql.password,
    "database": cfg.mysql.database,
    "charset": cfg.mysql.charset,
}


def main():
    conn = pymysql.connect(**MYSQL_CONFIG)
    cur = conn.cursor()

    # 检查 users 表是否存在
    cur.execute("SHOW TABLES LIKE 'users'")
    if cur.fetchone():
        print("users 表已存在，检查字段...")
        # 检查 role 字段是否存在
        cur.execute("SHOW COLUMNS FROM users LIKE 'role'")
        if not cur.fetchone():
            print("添加 role 字段...")
            cur.execute("ALTER TABLE users ADD COLUMN role ENUM('admin', 'user') DEFAULT 'user' AFTER password_hash")
        # 检查 email 字段
        cur.execute("SHOW COLUMNS FROM users LIKE 'email'")
        if not cur.fetchone():
            print("添加 email 字段...")
            cur.execute("ALTER TABLE users ADD COLUMN email VARCHAR(100) DEFAULT '' AFTER role")
        # 检查 avatar 字段
        cur.execute("SHOW COLUMNS FROM users LIKE 'avatar'")
        if not cur.fetchone():
            print("添加 avatar 字段...")
            cur.execute("ALTER TABLE users ADD COLUMN avatar VARCHAR(255) DEFAULT '' AFTER email")
        # 检查 status 字段
        cur.execute("SHOW COLUMNS FROM users LIKE 'status'")
        if not cur.fetchone():
            print("添加 status 字段...")
            cur.execute("ALTER TABLE users ADD COLUMN status ENUM('active', 'disabled') DEFAULT 'active' AFTER avatar")
        # 检查 updated_at 字段
        cur.execute("SHOW COLUMNS FROM users LIKE 'updated_at'")
        if not cur.fetchone():
            print("添加 updated_at 字段...")
            cur.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at")
        conn.commit()
        print("users 表字段更新完成")
    else:
        cur.execute("""
            CREATE TABLE users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('admin', 'user') DEFAULT 'user',
                email VARCHAR(100) DEFAULT '',
                avatar VARCHAR(255) DEFAULT '',
                status ENUM('active', 'disabled') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        conn.commit()
        print("users 表创建成功")

    # 检查默认管理员账号
    cur.execute("SELECT id FROM users WHERE username='admin'")
    if not cur.fetchone():
        # 创建默认管理员（密码: admin123）
        password_hash = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode("utf-8")
        cur.execute(
            "INSERT INTO users (username, password_hash, role, email, status) VALUES (%s, %s, %s, %s, %s)",
            ("admin", password_hash, "admin", "admin@finrag.com", "active"),
        )
        conn.commit()
        print("默认管理员账号创建成功: admin / admin123")
    else:
        print("默认管理员账号已存在")

    conn.close()


if __name__ == "__main__":
    main()

-- ============================================
-- FinRag MySQL 数据库初始化脚本
-- 适用于 Docker 部署（挂载到 /docker-entrypoint-initdb.d）
-- 也适用于本地手动执行
-- ============================================

CREATE DATABASE IF NOT EXISTS finrag_qa CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE finrag_qa;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'user') DEFAULT 'user',
    email VARCHAR(100) DEFAULT '',
    avatar VARCHAR(255) DEFAULT '',
    status ENUM('active', 'disabled') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 默认管理员账号（用户名: admin, 密码: admin123）
INSERT IGNORE INTO users (username, password_hash, role, email, status) VALUES
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYJRk5eKzby', 'admin', 'admin@finrag.com', 'active');

-- 高频问答对表
CREATE TABLE IF NOT EXISTS finance_faq (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    question VARCHAR(1000) NOT NULL,
    answer TEXT NOT NULL,
    source VARCHAR(200) DEFAULT '',
    type VARCHAR(50) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_category (category),
    KEY idx_question (question(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 知识库表
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    chunk_size INT DEFAULT 512,
    overlap INT DEFAULT 50,
    retrieval_type ENUM('dense','hybrid') DEFAULT 'hybrid',
    owner_id INT DEFAULT NULL,
    is_builtin TINYINT(1) DEFAULT 0,
    domain VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    doc_count INT DEFAULT 0,
    FOREIGN KEY (owner_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 文档表
CREATE TABLE IF NOT EXISTS documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    kb_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    status ENUM('uploaded', 'parsing', 'indexed', 'failed', 'deleted', 'cancelled') DEFAULT 'uploaded',
    chunk_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    KEY idx_kb_id (kb_id),
    KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 插入内置知识库
INSERT IGNORE INTO knowledge_bases (name, description, domain, is_builtin) VALUES
('banking 知识库', '商业银行、信贷、支付结算等银行业务知识', 'banking', 1),
('corporate_finance 知识库', '企业融资、并购重组、资本结构等公司金融知识', 'corporate_finance', 1),
('financial_accounting 知识库', '会计准则、财务报表、审计等财务会计知识', 'financial_accounting', 1);

-- 审计日志表（金融合规：登录、上传/删除文档、创建/删除知识库、管理员操作留痕）
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT DEFAULT NULL,
    username VARCHAR(50) DEFAULT '',
    action VARCHAR(50) NOT NULL,
    target_type VARCHAR(50) DEFAULT '',
    target_id VARCHAR(64) DEFAULT '',
    detail VARCHAR(1000) DEFAULT '',
    ip VARCHAR(64) DEFAULT '',
    request_id VARCHAR(64) DEFAULT '',
    success TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_action (action),
    KEY idx_user (user_id),
    KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

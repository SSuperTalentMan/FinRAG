import sys, requests
sys.path.insert(0, ".")
HOST = "http://127.0.0.1:8000"
BASE = f"{HOST}/api/v1"  # API 路由前缀

# 非路由端点（健康检查、首页）直接从 HOST 访问
for path in ["/", "/health", "/health/ready"]:
    try:
        r = requests.get(HOST + path, timeout=10)
        print("GET", path, "->", r.status_code, r.text[:150].replace("\n", " "))
    except Exception as e:
        print("GET", path, "ERR", str(e)[:150])

# API 路由
for path in ["/auth/me"]:
    try:
        r = requests.get(BASE + path, timeout=10)
        print("GET", path, "->", r.status_code, r.text[:150].replace("\n", " "))
    except Exception as e:
        print("GET", path, "ERR", str(e)[:150])

print("--- POST /chat/ (注册制) ---")
try:
    r = requests.post(
        BASE + "/chat/",
        json={"message": "注册制下发行人信息披露有哪些要求？", "session_id": "diag", "save_user": False},
        timeout=90,
    )
    print("POST /chat/ ->", r.status_code)
    print(r.text[:600].replace("\n", " "))
except Exception as e:
    print("POST /chat/ ERR", str(e)[:250])

"""跨区域算网作业安置服务的服务端包入口。"""

PROJECT_CODE = "compute_network_scheduler"


def project_info() -> dict[str, str]:
    """返回稳定的项目标识，供运行检查和诊断使用。"""
    return {"code": PROJECT_CODE, "title": "跨区域算网作业安置服务"}

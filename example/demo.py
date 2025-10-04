import os
import cv2
import json
import time
import base64
import argparse
import requests
import numpy as np
from pathlib import Path

# ========== 工具函数 ==========

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_to_image(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_base64_image(img_base64: str, save_path: str):
    """将 base64 编码的图片保存到本地文件"""
    save_path = Path(save_path)
    img = decode_base64_to_image(img_base64)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"✅ 已保存生成图片: {save_path}")


def draw_and_save_boxes(image_path: str, bboxes, save_path: str, color=(0, 255, 0), thickness: int = 2):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    for box in bboxes or []:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"✅ 检测框已绘制: {save_path}")


# ========== API 调用 ==========

TASK_ENDPOINTS = {
    "tapian_danzi_classify": "/tapian_danzi_classify/",
    "moben_danzi_classify": "/moben_danzi_classify/",
    "tapian_danzi_detect": "/tapian_danzi_detect/",
    "moben_danzi_detect": "/moben_danzi_detect/",
    "input_leixing_classify": "/input_leixing_classify/",
    "moben_poyi_classify": "/moben_poyi_classify/",
    "obsd_inference": "/obsd_inference/",
    "p3_inference": "/p3_inference/",
    "evobc_inference": "/evobc_inference/",
    "tapian_juzi_detect": "/tapian_juzi_detect/",
    # 非图片类任务（JSON 转发）
    "get_order_1": "/get_order_1/",
    "dino_search": "/dino_search/",
    "denoise_tapian_danzi": "/denoise_tapian_danzi/",
}


def post_image_base64(session: requests.Session, server_url: str, endpoint: str, image_b64: str, timeout: int = 60) -> dict:
    url = f"{server_url.rstrip('/')}{endpoint}"
    resp = session.post(url, json={"image": image_b64}, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"API 调用失败: {resp.status_code}, {resp.text[:500]}") from e
    return resp.json()


def call_task(session: requests.Session, task: str, image_path: str, server_url: str, timeout: int = 60) -> dict:
    if task not in TASK_ENDPOINTS:
        raise ValueError(f"不包含这个模型: {task}")
    img_b64 = encode_image_to_base64(image_path)
    return post_image_base64(session, server_url, TASK_ENDPOINTS[task], img_b64, timeout=timeout)


def call_moben_danzi_detect(image_path: str, server_url: str, save_dir: str, session: requests.Session) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    result = call_task(session, "moben_danzi_detect", image_path, server_url, timeout=90)

    # 打印检测框信息（若存在）
    results = result.get("results", {})
    # if results:
    #     print("检测结果信息：")
    #     print("bboxes:", results.get("bboxes", []))
    #     print("labels:", results.get("labels", []))
    #     print("scores:", results.get("scores", []))

    # 保存可视化整图
    viz_b64 = result.get("image")
    if viz_b64:
        boxed_path = os.path.join(save_dir, "output_with_boxes.png")
        save_base64_image(viz_b64, boxed_path)

    # 保存每个甲骨文裁剪图
    crops = result.get("cropped_jgw_images") or []
    for i, crop_b64 in enumerate(crops):
        crop_path = os.path.join(save_dir, f"jgw_crop_{i+1}.png")
        save_base64_image(crop_b64, crop_path)

    return result


def call_tapian_danzi_detect(image_path: str, server_url: str, save_dir: str, session: requests.Session) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    result = call_task(session, "tapian_danzi_detect", image_path, server_url, timeout=90)

    # 打印检测框信息（若存在）
    results = result.get("results", {})
    # if results:
    #     print("检测结果信息：")
    #     print("bboxes:", results.get("bboxes", []))
    #     print("labels:", results.get("labels", []))
    #     print("scores:", results.get("scores", []))

    # 保存可视化整图
    viz_b64 = result.get("image")
    if viz_b64:
        boxed_path = os.path.join(save_dir, "output_with_boxes.png")
        save_base64_image(viz_b64, boxed_path)

    # 保存每个甲骨文裁剪图
    crops = result.get("cropped_jgw_images") or []
    for i, crop_b64 in enumerate(crops):
        crop_path = os.path.join(save_dir, f"jgw_crop_{i+1}.png")
        save_base64_image(crop_b64, crop_path)

    return result

# ========== 演示运行 ==========

EXAMPLE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EXAMPLE_DIR / "images"

DEFAULT_IMAGES = {
    "moben_danzi_detect": str(IMAGES_DIR / "h00659.jpg"),
    "moben_poyi_classify": str(IMAGES_DIR / "保.png"),
    "input_leixing_classify": str(IMAGES_DIR / "h00026_190.jpg"),
    "tapian_danzi_classify": str(IMAGES_DIR / "h00026_190.jpg"),
    "moben_danzi_classify": str(IMAGES_DIR / "保.png"),
    "tapian_danzi_detect": str(IMAGES_DIR / "合52拓片.png"),
    "tapian_juzi_detect": str(IMAGES_DIR / "合52拓片.png"),
    "obsd_inference": str(IMAGES_DIR / "保.png"),
    "p3_inference": str(IMAGES_DIR / "保.png"),
    "evobc_inference": str(IMAGES_DIR / "保.png"),
    # get_order_1 不需要图片，占位即可
    "get_order_1": str(IMAGES_DIR / "保.png"),
    "dino_search": str(IMAGES_DIR / "安.png"),
    "denoise_tapian_danzi": str(IMAGES_DIR / "h00026_190.jpg"),
}


# ========== 非图片类任务：排序接口 ==========

def build_order_payload(col_threshold: float = 0.07) -> dict:
    """构造与 call_get_order_1.py 相同的示例请求体。"""
    categories = [1103, 966, 1540, 729, 914, 1006, 496, 336]
    positions = [
        [0.315, 0.352, 0.092, 0.045],
        [0.305, 0.417, 0.087, 0.088],
        [0.296, 0.489, 0.084, 0.079],
        [0.312, 0.569, 0.117, 0.109],
        [0.391, 0.373, 0.104, 0.078],
        [0.435, 0.523, 0.092, 0.139],
        [0.501, 0.372, 0.104, 0.126],
        [0.516, 0.545, 0.139, 0.120],
    ]
    return {
        "categories": categories,
        "positions": positions,
        "col_threshold": col_threshold,
    }


def call_get_order_1(session: requests.Session, server_url: str, col_threshold: float = 0.07, timeout: int = 60) -> dict:
    url = f"{server_url.rstrip('/')}{TASK_ENDPOINTS['get_order_1']}"
    payload = build_order_payload(col_threshold)
    resp = session.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"排序接口调用失败: {resp.status_code}, {resp.text[:500]}") from e
    return resp.json()


def choose_image(task: str, args) -> str:
    # 优先使用 --image-all 覆盖；否则按任务专项参数；否则默认
    if args.image_all:
        return args.image_all
    per_task = getattr(args, f"image_{task}", None)
    if per_task:
        return per_task
    return DEFAULT_IMAGES[task]


def run_demo(tasks, server_url: str, save_dir: str):
    start_time = time.time()
    timings: dict[str, float] = {}
    session = requests.Session()

    for task in tasks:
        if task == "get_order_1":
            print(f"\n▶ 运行任务: {task}")
            t0 = time.time()
            try:
                result = call_get_order_1(session, server_url, col_threshold=run_demo.args.order_col_threshold, timeout=90)
                # 输出简要结果
                short = json.dumps(result, ensure_ascii=False)
                short = short[:300] + ("..." if len(short) > 300 else "")
                print(f"{task} 结果片段: {short}")

                # 额外摘要
                sorted_indices = result.get("sorted_indices")
                sorted_categories = result.get("sorted_categories")
                if sorted_indices is not None:
                    print("摘要:")
                    print(f"  sorted_indices   = {sorted_indices}")
                    print(f"  sorted_categories= {sorted_categories}")
            except Exception as e:
                print(f"❌ 任务 {task} 失败: {e}")
                result = None
            finally:
                timings[task] = time.time() - t0
            continue

        # 其余任务按图片流程执行
        img_path = choose_image(task, run_demo.args)
        print(f"\n▶ 运行任务: {task} | 图片: {img_path}")
        t0 = time.time()
        try:
            if task == "moben_danzi_detect":
                result = call_moben_danzi_detect(img_path, server_url, os.path.join(save_dir, task), session)
            elif task == "tapian_danzi_detect":
                result = call_tapian_danzi_detect(img_path, server_url, os.path.join(save_dir, task), session)
            else:
                result = call_task(session, task, img_path, server_url)

                # 特殊处理：OBSD 结果包含 font_img
                if task == "obsd_inference":
                    font_img_b64 = result.get("results", {}).get("font_img")
                    if font_img_b64:
                        out_font = os.path.join(save_dir, task, "font_img.png")
                        save_base64_image(font_img_b64, out_font)
                
                if task == "denoise_tapian_danzi":
                    denoise_img_b64 = result.get("image")
                    if denoise_img_b64:
                        out_denoise = os.path.join(save_dir, task, "denoise_img.png")
                        save_base64_image(denoise_img_b64, out_denoise)

            # 输出简要结果
            result_json = json.dumps(result, ensure_ascii=False)
            short = result_json[:300] + ("..." if len(result_json) > 300 else "")
            print(f"{task} 结果片段: {short}")
        except Exception as e:
            print(f"❌ 任务 {task} 失败: {e}")
            result = None
        finally:
            timings[task] = time.time() - t0

    end_time = time.time()
    print("\n===== 汇总 =====")
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
    print("各任务耗时：")
    for k, v in timings.items():
        print(f"- {k}: {v:.2f} 秒")


def parse_args():
    parser = argparse.ArgumentParser(description="调用聚合 API 的演示脚本")
    parser.add_argument("--server-url", default=os.getenv("JICHENG_SERVER_URL", "http://vlrlabmonkey.xyz:7680"), help="API 服务器地址")
    parser.add_argument("--save-dir", default="example/outputs", help="输出保存目录")
    parser.add_argument("--tasks", nargs="*", choices=list(DEFAULT_IMAGES.keys()),
                        default=[
                            "moben_danzi_detect",
                            "moben_poyi_classify",
                            "input_leixing_classify",
                            "tapian_danzi_classify",
                            "moben_danzi_classify",
                            "tapian_danzi_detect",
                            "obsd_inference",
                            "p3_inference",
                            "evobc_inference",
                            "get_order_1",
                            "dino_search",
                            "denoise_tapian_danzi"
                        ], help="需要运行的任务列表")

    # 排序接口参数
    parser.add_argument("--order-col-threshold", type=float, default=0.07,
                        help="get_order_1 的分列阈值（默认 0.07）")

    # 全局图片覆盖
    parser.add_argument("--image-all", help="为所有任务统一指定图片")

    # 分任务图片路径（可选）
    for t, p in DEFAULT_IMAGES.items():
        parser.add_argument(f"--image-{t}", default=None, help=f"{t} 使用的图片路径，默认 {p}")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # 把 args 挂到函数上，便于 choose_image 访问
    run_demo.args = args
    run_demo(args.tasks, args.server_url, args.save_dir)


if __name__ == "__main__":
    main()

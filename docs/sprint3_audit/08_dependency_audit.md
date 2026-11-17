# 08 - KIỂM TOÁN DANH MỤC THƯ VIỆN & PHỤ THUỘC (DEPENDENCY AUDIT)
**Dự án:** AutoClaim-AI (`AI_CAR`)  
**Sprint:** 3 - Audit & Architecture Optimization  
**Mục tiêu kiểm toán:** Phân tích toàn bộ tệp `requirements.txt`, đối chiếu với mã nguồn thực tế, phát hiện xung đột và thư viện dư thừa.

---

## 1. Bảng Đối Chiếu Thư Viện Trong `requirements.txt` và Mã Nguồn Thực Tế

| Thư viện khai báo | Phiên bản yêu cầu | Phiên bản cài thực tế | Vị trí sử dụng trong mã nguồn | Đánh giá mức độ cần thiết |
| :--- | :---: | :---: | :--- | :---: |
| `ultralytics` | `==8.4.178` | 8.4.178 | `pipeline.py`, `train_damage.py`, `export_onnx.py` | **BẮT BUỘC (Core)** |
| `torch` | `==2.14.0` | 2.6.0+cu124 | Toàn bộ backend deep learning | **BẮT BUỘC (Core)** |
| `torchvision` | `==0.29.0` | 0.20.0 | Phụ trợ cho Ultralytics và biến đổi tensor | **BẮT BUỘC (Core)** |
| `opencv-python` | `==5.0.0.93` | 5.0.0.93 | `cv2`: Đọc ảnh, SIFT, RANSAC Homography, vẽ box | **BẮT BUỘC (Core)** |
| `numpy` | `==2.3.5` | 2.3.5 | Thao tác ma trận, tọa độ polygon | **BẮT BUỘC (Core)** |
| `PyYAML` | `==6.0.3` | 6.0.3 | Đọc cấu hình `classes.yaml`, `damage.yaml` | **BẮT BUỘC (Core)** |
| `fastapi` | `==0.141.1` | 0.141.1 | `ai/api/main.py` (REST API Gateway) | **CẦN THIẾT** |
| `uvicorn` | `==0.52.4` | 0.52.4 | ASGI Web Server chạy FastAPI | **CẦN THIẾT** |
| `python-multipart` | `==0.0.32` | 0.0.32 | Hỗ trợ upload ảnh form-data qua API | **CẦN THIẾT** |
| `pytest` | `>=8.0.0` | 9.2.0 | Test runner cho `ai/tests/test_pipeline_unit.py` | **CẦN THIẾT** |
| `httpx` | `>=0.27.0` | 0.28.1 | Phục vụ kiểm thử async client cho FastAPI | **CẦN THIẾT** |
| `pillow` | `>=10.0.0` | 11.1.0 | Vẽ font chữ tiếng Việt Unicode trong `visualizer` | **CẦN THIẾT** |
| `shapely` | `>=2.0.0` | 2.0.7 | Tính toán hình học polygon giao nhau trong pipeline | **CẦN THIẾT** |
| `supervision` | `>=0.31.0` | 0.31.7 | Tính ma trận IoU `sv.box_iou_batch` trong đối soát | **CẦN THIẾT** |
| `onnx` | `>=1.23.0` | 1.23.2 | Xuất mô hình ONNX trong `export_onnx.py` | **CẦN THIẾT** |
| `onnxruntime` | `>=1.31.0` | 1.31.1 | Chạy suy luận mô hình ONNX độc lập | **CẦN THIẾT** |
| `pydantic` | `>=2.14.0` | 2.14.1 | Định nghĩa dữ liệu đầu vào/ra trong `schemas.py` | **CẦN THIẾT** |
| `graphifyy` | `>=0.9.100` | 0.9.107 | Trích xuất Knowledge Graph kiến trúc dự án | **CÔNG CỤ PHỤ TRỢ** |

---

## 2. Phát Hiện Xung Đột & Cảnh Báo Phiên Bản (Dependency Conflicts)

### 2.1. Lỗi Khai báo Phiên bản Ảo trong `requirements.txt`
- Trong `requirements.txt` có ghi: `torch==2.14.0` và `torchvision==0.29.0`.
- **Thực tế kỹ thuật**: PyTorch chính thức chưa từng phát hành phiên bản `2.14.0` (sau bản 2.1 là 2.2, 2.3... và hiện tại là 2.6.0). Phiên bản thực tế đang chạy trên máy là `torch 2.6.0+cu124`. Khai báo `torch==2.14.0` sẽ khiến các thành viên khác khi chạy `pip install -r requirements.txt` trên máy mới bị báo lỗi `Could not find a version that satisfies the requirement`.
$\Rightarrow$ **Đề xuất:** Sửa thành `torch>=2.4.0` và `torchvision>=0.19.0`.

### 2.2. Xung đột Sympy giữa Torch và Onnxslim
- Khi Ultralytics xuất ONNX và tự động cài `onnxslim`, thư viện này đã kéo theo `sympy 1.14.0`, dẫn đến cảnh báo xung đột từ PyTorch:
  `torch 2.6.0+cu124 requires sympy==1.13.1, but you have sympy 1.14.0`.
- *Mức độ nguy hại*: Hiện chưa làm ảnh hưởng suy luận, nhưng cần ghim `sympy==1.13.1` trong môi trường sản xuất.

### 2.3. Cảnh báo CUDA Execution Provider trong ONNX Runtime
- Khi nạp mô hình ONNX, do máy có card đồ họa NVIDIA RTX 4050, `onnxruntime` tự động dò tìm `CUDAExecutionProvider` nhưng thiếu file `cublasLt64_13.dll` (của CUDA 13).
- ONNX Runtime đã tự động fallback về `CPUExecutionProvider` an toàn. Để tránh xuất hiện log cảnh báo màu đỏ dài hàng chục dòng trên terminal, trong code chỉ định rõ:
  `providers=['CPUExecutionProvider']`.

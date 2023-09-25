# Cài đặt các thư viện cần thiết bằng cách chạy lệnh:

```shell
 pip install -r requirements.txt
```

- **Chạy tệp [main.py](./main.py)**

## Thay đổi bản đồ và các tham số khác trong mã nguồn.

- **Để thêm chướng ngại vật động:**
  **Thêm các tham số sau vào `dynamic_obs_list`:**

  - `(x, y)` - vị trí bắt đầu của chướng ngại vật động.
  - `(width, height)` - kích thước của chướng ngại vật động.
  - Hướng di chuyển của chướng ngại vật động (`1 - lên, 2 - trái, 3 - phải, 4 - xuống`).
  - Tốc độ của chướng ngại vật động.

- **Để chỉnh sửa bản đồ:**
  - **_Nhấp chuột trái_** để thay đổi trạng thái của một ô (ô trống -> chướng ngại vật tĩnh và ngược lại).
  - **_Nhấp chuột phải_** để thay đổi vị trí bắt đầu của robot.
  - **_Để thay đổi tốc độ chạy:_** sử dụng các phím mũi tên trái và phải.

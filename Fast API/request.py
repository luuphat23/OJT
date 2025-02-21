import requests

# Địa chỉ của API (đảm bảo rằng bạn đang sử dụng cổng 8001)
url = "http://localhost:8001/predict/"

# Đường dẫn đến file ảnh bạn muốn gửi
file_path = r"C:\Users\ADMIN\test.jpg"  # Thay đổi đường dẫn đến file ảnh của bạn

# Mở file ảnh và gửi yêu cầu
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Kiểm tra mã trạng thái và in ra nội dung phản hồi
print("Status Code:", response.status_code)

if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Response Text:", response.text)  # In ra nội dung phản hồi để kiểm tra

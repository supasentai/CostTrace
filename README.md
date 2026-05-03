# **CostTrace: Mô phỏng Can thiệp Thích ứng có Ràng buộc Ngân sách trên Mạng lưới Dịch tễ**

## **Giới thiệu**

Dự án này cung cấp một framework tính toán và mô phỏng nhằm tối ưu hóa chiến lược cách ly trong mạng lưới tiếp xúc dịch tễ (Epidemic Contact Networks). Hệ thống giải quyết khoảng trống nghiên cứu của các mô hình truy vết SOTA hiện tại (như DeepTrace) bằng cách chuyển từ bài toán nhận diện nguồn lây sang bài toán can thiệp có ràng buộc (Budget-Constrained Intervention). Dự án được triển khai bằng Python, kết hợp các thuật toán lý thuyết đồ thị cổ điển và Graph Neural Networks (GNN) để đánh giá hiệu quả giảm thiểu đỉnh dịch dưới các mức giới hạn tài nguyên y tế cụ thể.

## **Kiến trúc hệ thống**

1. **Tiền xử lý dữ liệu (Data Processing):** Xử lý bộ dữ liệu tiếp xúc vật lý thực tế (SocioPatterns hoặc dữ liệu truy vết COVID-19) thành cấu trúc ma trận kề (Adjacency Matrix) và danh sách liên kết (Edge List).  
2. **Đánh giá rủi ro (Risk Scoring):** Áp dụng mô hình GraphSAGE (hoạt động như DeepTrace Proxy) và các thuật toán Centrality (Degree, Betweenness, PageRank) để định lượng mức độ nguy hiểm lây nhiễm cho từng đỉnh (Node).  
3. **Tối ưu hóa can thiệp (Budget-Constrained Isolation):** Trích xuất và loại bỏ Top\-![][image1] các node có điểm rủi ro cao nhất khỏi cấu trúc đồ thị dựa trên ràng buộc ngân sách thiết lập trước (ví dụ: ![][image2]).  
4. **Mô phỏng Dịch tễ (Epidemic Simulation):** Khởi chạy mô hình lan truyền SIR thông qua thư viện EoN trên đồ thị đã can thiệp để mô phỏng quỹ đạo lây nhiễm.  
5. **Trực quan hóa (Visualization):** Đánh giá hiệu suất thông qua biểu đồ đường (Infection Curves) so sánh đỉnh dịch (Peak Infection) giữa các chiến lược và sử dụng NetworkX/Gephi để kết xuất cấu trúc mạng lưới.

`requirements.txt`
```
numpy  
pandas  
matplotlib  
networkx  
torch  
torch\_geometric  
EoN
```
## **Cài đặt và triển khai**

1. **Thiết lập môi trường ảo và cài đặt thư viện:**

```
python \-m venv venv    
\# Trên Linux/macOS    
source venv/bin/activate      
\# Trên Windows    
venv\\Scripts\\activate

pip install \-r requirements.txt
```
2. **Khởi chạy luồng mô phỏng chính:** Đảm bảo bạn đang ở thư mục gốc của dự án, thực thi tập lệnh sau để chạy đường ống phân tích:
```
python main.py \--dataset sociopatterns \--budget 0.05 \--epochs 200
```
3. **Thực thi và đối chiếu:** \* Hệ thống sẽ tự động tính toán Centrality, huấn luyện GraphSAGE proxy và chạy mô phỏng SIR.  
* Các tệp kết quả (.csv chứa số liệu thống kê và .png chứa biểu đồ lây nhiễm) sẽ được tự động xuất vào thư mục results/.

## **Hướng phát triển và tối ưu**

* **Lọc trạng thái miễn dịch (Immunity Filtering):** Tích hợp kiểm tra trạng thái phục hồi (Recovered) của các cá nhân trước khi đưa vào danh sách cách ly. Việc này giúp tránh phân bổ ngân sách can thiệp lãng phí vào những node đã có kháng thể tự nhiên.  
* **Đồ thị động thời gian \- không gian (Spatio-Temporal Networks):** Mở rộng kiến trúc hệ thống để xử lý đồ thị động (Dynamic Graphs), trong đó các cạnh liên kết chỉ được kích hoạt trong những khung thời gian (time-windows) cụ thể, phản ánh chính xác độ trễ của chuỗi lây nhiễm trong thực tế.

## **Tài liệu tham khảo**

1. Tan, C. W., Yu, P. D., Chen, S., & Poor, H. V. (2025). DeepTrace: Learning to Optimize Contact Tracing in Epidemic Networks with Graph Neural Networks. *IEEE Transactions on Signal and Information Processing over Networks*. https://arxiv.org/abs/2211.00880  
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems* (pp. 1024-1034).  
3. Kiss, I. Z., Miller, J. C., & Simon, P. L. (2017). *Mathematics of Epidemics on Networks: From Exact to Approximate Models*. Springer.  
4. Adamopoulos, I., Valamontes, A., Syrou, N., Adamopoulou, J., & Bardavouras, A. (2025). Utilizing Graph Theory Algorithms for the Modeling and Analysis of COVID-19 Infection Dynamics. *Mesopotamian Journal of Artificial Intelligence in Healthcare*, 2025, 1-11.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAYCAYAAAALQIb7AAACeElEQVR4Xu2VO2hTURjHc0kLirYoEmPJ49ybRENxaQkKgqMFRQRRF6kgCNJFQdtBcWkdHHzgIAh1a3HQobQUdFAX0Um6ODiIUGhB3OrgqGD9/XvPuZ4cTDu65A8f53u/zrlJLteFh0KhsLNer++FzYc2D/lms9kXKn1EpVJpTxzHu0KDgzHmDPQeugXN47sv9AF59Dexj4eGDWC8i/EntN7JiWkq2D5Wq9WjkjkvIH8ndqpcLu+HH4A/Dr3G9kobCHNksF3/cslCYDsBrTJ9WTJ+NeSHUAJ/jnOUQqc1MTQcxrcBh0fQCl2WQptAwhsqpgkk20lm/QmQx0yHzWTQZZr0Ll4SsC20CyadPCumCWlg2vnDH8T2dNP1CTgNQmvq3qqiWq12AHmESbdLkSRJE5/PlUrlkGR7P9fFywffmS3XJ+A0Cv0m4Fir1eolyW3k+5yL3loi+Al07zgvQS/c/cnH89scxt4XXdcpOEnCIZtgPUzCFA18TjUajX4bO4z83K3PbuAeOc6rcT9Wu96NcQn6hMMTzkHptUatqVgs7mgL8KACKqSCkil0GPmDbeAy8ddQR1mAkkNr0A9NpxW6rreCnd5N3gP/DLojwd7jdNvrNt73ZVf0BXmh06t0sN376yuiW/YembZ2FWr5QW3fF0VmkZfsek8iX8ycLcL1CfAD0KpfDPkKdCQLQngLzSH2SGeLzWkNnA+gxAU7oBvHb8zXuW/VK6bXq/iNN6DR4c03deCC4M9CX036syP93wtO7VrfjPv+fKgQ9gW9QnslurPML9K6csHfhSaO//Hrr3sk2WMlCm2CEsfpj7p+jd7AD4U+XXTx//EH0g2jNqj1wesAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>

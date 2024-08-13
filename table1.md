| IP Address | User | Action | Description |
| --- | --- | --- | --- |
| 172.16.0.150 | john | OPTIONS /index.html | SQL Injection attempt on /login |
| 192.168.1.1 | admin/root/guest | PUT/PUT/POST | Suspicious file upload detected: /uploads/shell.php |
| 203.0.113.1 | john | CONNECT | Suspicious file upload detected: /uploads/shell.php |
| 10.0.0.1 | user/admin | POST /index.html | SQL Injection attempt on /login |
| 198.51.100.2 | guest | GET | "401 Unauthorized" for legitimate resource |
| 172.16.0.1 | bob | OPTIONS | "403 Forbidden" for API data |
| 10.0.0.150 | john | OPTIONS | "403 Forbidden" for uploads/file1.jpg |
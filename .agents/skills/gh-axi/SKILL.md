---
name: gh-axi
description: "Thao tác với GitHub qua CLI gh-axi tối ưu token (TOON format): issues, PRs, workflow runs, releases, repos, secrets, search. Ưu tiên dùng gh-axi thay vì gh thông thường khi làm việc với GitHub."
user-invocable: false
author: Kun Chen (kunchenguid)
---

# gh-axi — Agent-Ergonomic GitHub CLI

Công cụ CLI wrapper bọc quanh GitHub CLI (`gh`), thiết kế theo chuẩn **AXI (Agent eXperience Interface)** giúp Agent tiết kiệm token (định dạng TOON), không bị treo bởi interactive prompts, và xử lý kết quả chính xác.

## 1. Yêu cầu Tiền đề (Prerequisites)

- Đã cài đặt `gh` (`v2.102.0+`) và `gh-axi` (`0.1.35+`) tại `/home/hongphuoc/.local/bin`.
- Cần đăng nhập GitHub:
  - Hoặc chạy `gh auth login`
  - Hoặc set biến môi trường `GITHUB_TOKEN`

## 2. Cách Tra Cứu & Sử Dụng (Source of Truth)

Không tra cứu cờ lệnh cũ trong tài liệu tĩnh, luôn gọi trực tiếp CLI:

- Dashboard repository hiện tại: `gh-axi`
- Danh mục lệnh và cờ toàn cục: `gh-axi --help`
- Chi tiết từng lệnh: `gh-axi <command> --help`

## 3. Các Lệnh Thường Dùng

- **Issues:** `gh-axi issue list --state open`, `gh-axi issue view <id>`, `gh-axi issue create`
- **Pull Requests:** `gh-axi pr list`, `gh-axi pr view <id>`, `gh-axi pr diff <id>`
- **CI / Workflows:** `gh-axi run list`, `gh-axi run view <id>`, `gh-axi workflow list`
- **Releases & Repos:** `gh-axi release list`, `gh-axi repo view`

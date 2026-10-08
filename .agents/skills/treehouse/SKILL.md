---
name: treehouse
description: "Quản lý Git worktree cho các workflow đa Agent song song: tạo, khóa (lock), đồng bộ, merge và dọn dẹp worktree độc lập."
user-invocable: true
---

# Treehouse — Git Worktree Manager for AI Agents

Công cụ quản lý Git Worktree tối ưu cho các workflow chạy đa Agent song song, hỗ trợ cả giao diện CLI và MCP server.

## 1. Cấu hình Dự án

- File cấu hình: `treehouse.json`
- Thư mục worktree mặc định: `.worktrees` (đã khai báo trong `.gitignore`)
- MCP Server: `treehouse-mcp` (được cấu hình trong `~/.gemini/config/mcp_config.json`)

## 2. Quy trình Làm việc với Subagents / Song song

Khi cần phân chia tác vụ cho subagent hoặc làm việc trên nhánh độc lập:

1. **Tạo worktree và lock cho Agent:**
   ```bash
   treehouse create <tên-worktree> [tên-branch] --agent <agent-id> --message "<mô tả tác vụ>"
   ```
   *Ví dụ:* `treehouse create exp-model-b feature/model-b --agent subagent-1 --message "Training Strategy B"`

2. **Kiểm tra trạng thái:**
   ```bash
   treehouse list
   treehouse status [tên-worktree]
   ```

3. **Mở khóa (khi cần chuyển giao):**
   ```bash
   treehouse unlock <tên-worktree>
   ```

4. **Hoàn tất và Merge:**
   ```bash
   # Merge vào nhánh hiện tại và xóa branch
   treehouse complete <tên-worktree> --merge --delete-branch
   
   # Hoặc squash merge với commit message
   treehouse complete <tên-worktree> --merge --squash --message "feat: integrate model B"
   ```

5. **Dọn dẹp:**
   ```bash
   treehouse remove <tên-worktree>
   treehouse prune
   ```

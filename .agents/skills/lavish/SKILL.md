---
name: lavish
description: Turn complex, architectural, or visual responses into rich, interactive, reviewable HTML artifacts using the lavish-axi CLI. Use when presenting project plans, pipeline architectures, data exploration, comparisons, diagrams, or reports where visual interactive review and user annotation in a local browser are better than text.
license: MIT
metadata:
  author: Kun Chen (kunchenguid)
  hermes-tags: html, review, artifacts, visualization, lavish-axi
  hermes-category: productivity
---

# Lavish Editor (AXI)

Lavish Editor opens agent-generated HTML in the user's local browser so a human can inspect, annotate elements/text, edit whiteboard diagrams, and send structured feedback directly to the agent.

## Core Workflows

### 1. Tạo và mở Artifact HTML
Tạo file HTML (khuyên dùng thư mục `.lavish/` trong thư mục dự án), sau đó chạy:
```bash
lavish-axi <path-to-html>
```
Lệnh sẽ mở một máy chủ cục bộ (local express server) và hiển thị artifact trên trình duyệt.

### 2. Nhận phản hồi người dùng (Long Poll)
Khi cần lắng nghe phản hồi và ghi chú (annotation) từ người dùng trên giao diện web:
```bash
lavish-axi poll <path-to-html>
```

### 3. Phản hồi cho người dùng qua giao diện
```bash
lavish-axi reply <path-to-html> --agent-reply "<thông điệp>"
```

### 4. Xuất file tĩnh (Export)
Tạo file HTML độc lập (inline assets cục bộ) để chia sẻ mà không cần máy chủ lavish:
```bash
lavish-axi export <path-to-html> --out <path-to-output.html>
```

### 5. Dừng server
```bash
lavish-axi stop
```

## Hướng dẫn thiết kế giao diện (Design System)
Ưu tiên sử dụng CDN Tailwind CSS v4 kết hợp DaisyUI v5 để tạo giao diện hiện đại, trực quan:
```html
<!DOCTYPE html>
<html lang="vi" data-theme="corporate">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tiêu đề</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5" type="text/css" />
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-base-200 min-h-screen p-6">
  <!-- Nội dung tương tác -->
</body>
</html>
```

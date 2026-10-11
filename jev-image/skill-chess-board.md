---
name: "chess-board-renderer"
description: "使用 HTML/CSS/JS 渲染专业的中国象棋棋盘。提供可复用的坐标系统、线条绘制函数和完整的棋盘布局逻辑，包括米字格（宫殿对角线）、楚河汉界（河流）和棋子定位。"
---

# 中国象棋棋盘渲染器

使用 HTML、CSS 和 JavaScript 渲染专业的中国象棋棋盘。提供可复用的坐标系统、线条绘制函数和完整的棋盘布局逻辑。

## 使用场景

当需要以下功能时使用此技能：
- 绘制或渲染中国象棋棋盘
- 实现象棋游戏界面
- 创建符合象棋规范的棋盘
- 绘制带对角线的网格（米字格）
- 绘制带河流分割线的棋盘（楚河汉界）

## 核心原则

### 1. 双层坐标系统

**始终使用两个坐标系统：**

```
┌─────────────────────────────────────────┐
│  网格坐标 (0-8, 0-9)                    │
│  - 用于游戏逻辑                         │
│  - 便于棋子定位                         │
│  - 易于理解                             │
│                                         │
│  像素坐标         │
│  - 用于渲染                             │
│  - DOM 定位                             │
│  - 从网格坐标计算得出, 注意棋子的中心要放在网格的交叉点上，所以要注意棋子大小的偏移量的计算                   │
└─────────────────────────────────────────┘
```

**坐标转换函数：**
```javascript
// 将网格坐标转换为像素坐标
function gridToPixel(gridX, gridY) {
    return {
        x: CONFIG.offsetX + gridX * CONFIG.gridSize,
        y: CONFIG.offsetY + gridY * CONFIG.gridSize
    };
}
```


### 2. 线条绘制函数

为每种线条类型创建专用函数。**永远不要硬编码坐标。**

```javascript
// 竖线：从 (gridX, gridY1) 到 (gridX, gridY2)
function drawVerticalLine(gridX, gridY1, gridY2) {
    const start = gridToPixel(gridX, gridY1);
    const end = gridToPixel(gridX, gridY2);
    
    const line = document.createElement('div');
    line.className = 'line';
    line.style.width = '2px';
    line.style.height = `${end.y - start.y}px`;
    line.style.left = `${start.x - 1}px`;
    line.style.top = `${start.y}px`;
    
    document.getElementById('board').appendChild(line);
}

// 横线：从 (gridX1, gridY) 到 (gridX2, gridY)
function drawHorizontalLine(gridY, gridX1, gridX2) {
    const start = gridToPixel(gridX1, gridY);
    const end = gridToPixel(gridX2, gridY);
    
    const line = document.createElement('div');
    line.className = 'line';
    line.style.width = `${end.x - start.x}px`;
    line.style.height = '2px';
    line.style.left = `${start.x}px`;
    line.style.top = `${start.y - 1}px`;
    
    document.getElementById('board').appendChild(line);
}

// 对角线：从 (gridX1, gridY1) 到 (gridX2, gridY2)
function drawDiagonal(gridX1, gridY1, gridX2, gridY2) {
    const start = gridToPixel(gridX1, gridY1);
    const end = gridToPixel(gridX2, gridY2);
    
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    
    const line = document.createElement('div');
    line.className = 'line';
    line.style.width = `${length}px`;
    line.style.height = '2px';
    line.style.left = `${start.x}px`;
    line.style.top = `${start.y}px`;
    line.style.transformOrigin = '0 50%';
    line.style.transform = `rotate(${angle}deg)`;
    
    document.getElementById('board').appendChild(line);
}

// 矩形块：从 (gridX1, gridY1) 到 (gridX2, gridY2) - 填充矩形区域
function drawBlock(gridX1, gridY1, gridX2, gridY2, backgroundColor, zIndex) {
    const start = gridToPixel(gridX1, gridY1);
    const end = gridToPixel(gridX2, gridY2);
    
    const width = end.x - start.x;
    const height = end.y - start.y;
    
    const block = document.createElement('div');
    block.className = 'block';
    block.style.position = 'absolute';
    block.style.width = `${width}px`;
    block.style.height = `${height}px`;
    block.style.left = `${start.x}px`;
    block.style.top = `${start.y}px`;
    block.style.backgroundColor = backgroundColor;
    block.style.zIndex = zIndex;
    
    document.getElementById('board').appendChild(block);
}
```

### 3. 绘制顺序

按以下顺序绘制元素：
1. **网格线**（竖线和横线）- 基础层
2. **米字格**（宫殿对角线）- 仅在宫殿区域
3. **河流**（楚河汉界）- 覆盖层
4. **棋子** - 顶层

## 完整棋盘布局

### 标准棋盘尺寸
- 9 条竖线（列）
- 10 条横线（行）

### 米字格（宫殿对角线）

**位置：** 在每个宫殿内（九宫格）
- 上宫：第 0-2 行，第 3-5 列
- 下宫：第 7-9 行，第 3-5 列

**仅 4 条对角线：**
```javascript
// 上宫：士 (3,0) 和 士 (5,2) → 將 (4,0)
drawDiagonal(3, 0, 5, 2);  // 左士 → 将
drawDiagonal(5, 0, 3, 2);  // 右士 → 将

// 下宫：仕 (3,9) 和 仕 (5,7) → 帥 (4,9)
drawDiagonal(3, 9, 5, 7);  // 左仕 → 帅
drawDiagonal(5, 9, 3, 7);  // 右仕 → 帅
```

**重要：** 米字格连接士/仕 到 將/帥，不是到兵/卒！

### 楚河汉界（河流）

**必须使用 drawRiver() 函数。函数自已写，请参考下面的函数签名要求，不要硬编码河流 HTML 或手动定位。**

函数签名：
```javascript
function drawRiver
   河流覆盖网格坐标 (0, 4) 到 (8, 5)
   使用 drawBlock() 创建河流区域

**行为和要求：**
- 覆盖网格坐标 (0, 4) 到 (8, 5) — 恰好 9 列 × 1 行
- 创建填充块，颜色与棋盘背景匹配
- 不超出棋盘边界
- 使用 drawBlock() 函数和网格坐标（不硬编码）需传入棋盘背景一样的backgroundColor, 和比棋盘背景大的zIndex，这样才能完美盖住棋盘多余的线来形成楚河汉界
- 添加文字 "楚河 漢界" 一次，居中， 不折行，适当的字母间距，需要放在，需要放在drawBlock函数画的那个块里。

## 棋子定位

### 初始棋子布局，注意上方和下方的棋子是分别不同的颜色

```javascript
const INITIAL_PIECES = {
    black: [  // 上方（北），从右到左读
        {name: '車', x: 0, y: 0}, {name: '馬', x: 1, y: 0}, {name: '象', x: 2, y: 0}, {name: '士', x: 3, y: 0},
        {name: '將', x: 4, y: 0}, {name: '士', x: 5, y: 0}, {name: '象', x: 6, y: 0}, {name: '馬', x: 7, y: 0}, {name: '車', x: 8, y: 0},
        {name: '炮', x: 1, y: 2}, {name: '炮', x: 7, y: 2},
        {name: '卒', x: 0, y: 3}, {name: '卒', x: 2, y: 3}, {name: '卒', x: 4, y: 3}, {name: '卒', x: 6, y: 3}, {name: '卒', x: 8, y: 3}
    ],
    red: [    // 下方（南），从左到右读
        {name: '車', x: 0, y: 9}, {name: '馬', x: 1, y: 9}, {name: '相', x: 2, y: 9}, {name: '仕', x: 3, y: 9},
        {name: '帥', x: 4, y: 9}, {name: '仕', x: 5, y: 9}, {name: '相', x: 6, y: 9}, {name: '馬', x: 7, y: 9}, {name: '車', x: 8, y: 9},
        {name: '砲', x: 1, y: 7}, {name: '砲', x: 7, y: 7},
        {name: '兵', x: 0, y: 6}, {name: '兵', x: 2, y: 6}, {name: '兵', x: 4, y: 6}, {name: '兵', x: 6, y: 6}, {name: '兵', x: 8, y: 6}
    ]
};
```

### 创建棋子函数

```javascript
function createPiece(name, color, gridX, gridY) {
    const pos = gridToPixel(gridX, gridY);
    
    const piece = document.createElement('div');
    piece.className = `piece ${color}`;
    piece.textContent = name;
    piece.dataset.name = name;
    piece.dataset.color = color;
    piece.dataset.x = gridX;
    piece.dataset.y = gridY;
    piece.style.left = `${pos.x}px`;
    piece.style.top = `${pos.y}px`;
    
    piece.addEventListener('click', (e) => {
        e.stopPropagation();
        handleClick(piece);
    });
    
    return piece;
}
```

## 实现检查清单
- [ ] 定义 CONFIG（gridSize、offsetX、offsetY）
- [ ] 实现 gridToPixel() 坐标转换函数
- [ ] 创建 drawVerticalLine(gridX, gridY1, gridY2)
- [ ] 创建 drawHorizontalLine(gridY, gridX1, gridX2)
- [ ] 创建 drawDiagonal(gridX1, gridY1, gridX2, gridY2)
- [ ] 创建 drawBlock(gridX1, gridY1, gridX2, gridY2, backgroundColor) - 用于绘制矩形区域，注意背景区域和棋盘的关系，设置合理的padding和margin
- [ ] 创建 drawRiver() 函数 - 必须使用 drawBlock() 绘制河流区域
- [ ] 绘制 9 条竖线（0-8）
- [ ] 绘制 10 条横线（0-9）
- [ ] 绘制 4 条米字格对角线（士/仕 → 將/帥）
- [ ] **调用 drawRiver()** - 必须使用此函数和 drawBlock()，不要硬编码！
- [ ] 在正确的网格位置放置 32 个棋子
- [ ] 设置棋子样式（红色棋子红色边框，黑色棋子黑色边框）

## 总结

1. **始终使用网格坐标**（0-8, 0-9）用于逻辑
2. **始终转换为像素**用于渲染
3. **创建线条绘制函数**接受网格坐标
4. **米字格 = 4 条对角线**连接士/仕 到 將/帥
5. **楚河汉界 = 必须调用 drawRiver()** - 使用网格 (0,4) 到 (8,5)，创建不透明块
6. **省略炮位框/兵位框** - 不是标准
7. **单独设置样式**基于用户需求

使用前面的技能输出html

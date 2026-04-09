# ReadTheDocs 本地验证说明

本文档用于指导如何在本地验证 ReadTheDocs 文档构建，**不会发布到 ReadTheDocs 网站**。

## 环境准备

确保已安装 Python 3.8+ 和 pip。

## 安装依赖

```bash
cd <项目根目录>
pip install -r docs/requirements.txt
```

> 示例：如果项目位于 `D:\code\msagent`，则执行 `cd D:\code\msagent`

## 本地构建

```bash
# 构建 HTML 文档
sphinx-build -b html docs/ docs/_build/html/

# 或者使用 make 命令（Linux/Mac）
make -C docs html
```

## 查看结果

构建完成后，在浏览器中打开：

```
<项目根目录>/docs/_build/html/index.html
```

或者在终端中运行（需要安装 Python 3 的 http 模块）：

```bash
# 进入构建目录
cd docs/_build/html/

# Python 3
python -m http.server 8000

# 然后访问 http://localhost:8000/
```

> 示例：如果项目位于 `D:\code\msagent`，则打开 `D:\code\msagent\docs\_build\html\index.html`

## 常见问题

### 1. 缺少依赖

如果遇到 `ModuleNotFoundError`，请检查 `docs/requirements.txt` 并安装所有依赖：

```bash
pip install -r docs/requirements.txt
```

### 2. 构建警告

Sphinx 可能会输出一些警告，这些通常不影响构建结果。如果看到错误（ERROR），则需要修复。

### 3. 中文显示问题

确保 `docs/conf.py` 中设置了 `language = 'zh_CN'`，并且系统支持中文字体。

## 清理构建文件

```bash
# Windows
rmdir /s /q docs\_build

# Linux/Mac
rm -rf docs/_build
```

## 推送到 ReadTheDocs

本地验证通过后，推送代码到远程仓库，ReadTheDocs 会自动检测代码更新并重新构建文档。

## 参考链接

- [ReadTheDocs 配置文档](https://docs.readthedocs.io/en/stable/config-file/v2.html)
- [Sphinx 文档](https://www.sphinx-doc.org/)

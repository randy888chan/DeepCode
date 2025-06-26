# 🐳 Docker环境宿主机代理配置 - 完整解决方案

## 📋 **问题背景**

在香港大学环境中：
- **宿主机**: 运行着代理服务 `http://127.0.0.1:7890`
- **Docker容器**: 需要通过宿主机代理访问AI服务以解决地区限制

## 🔑 **核心概念**

### **地址映射关系**
```
宿主机环境:
  代理地址: http://127.0.0.1:7890 ✅ 可用

Docker容器环境:
  宿主机访问: http://172.17.0.1:7890 ⚠️ 需要端口转发
  容器地址: 127.0.0.1 ❌ 指向容器自己
```

## ⚙️ **解决方案**

### **1. 自动配置系统** (`env_config.py`)

已实现智能代理配置：
```python
# 当前配置
DEFAULT_ENV_VARS = {
    "USE_PROXY": "True",      # 启用代理
    "AUTO_PROXY": "True",     # 智能检测
    "PROXY_HOST": "172.17.0.1",  # Docker内访问宿主机
    "PROXY_PORT": "7890",     # 代理端口
}
```

**行为逻辑**:
1. 优先尝试使用 `http://172.17.0.1:7890`
2. 如果不可达，自动降级到直连
3. 提供详细状态信息

### **2. 手动控制工具** (`proxy_control.py`)

快速管理代理设置：

```bash
# 查看状态
python proxy_control.py status

# 测试代理连接
python proxy_control.py test

# 启用代理（智能模式）
python proxy_control.py enable

# 强制启用代理
python proxy_control.py enable --force

# 禁用代理
python proxy_control.py disable

# 测试API访问
python proxy_control.py api
```

### **3. 启动脚本** (`start_with_network_config.sh`)

智能启动流程：
```bash
./start_with_network_config.sh
```

## 🚀 **使用步骤**

### **步骤1: 确认宿主机代理状态**

在宿主机上验证：
```bash
# 在宿主机执行
curl --proxy http://127.0.0.1:7890 http://httpbin.org/ip
```
应返回代理后的IP地址。

### **步骤2: 配置Docker网络访问**

如果代理端口不可达，可能需要配置Docker网络：

#### **方法A: 使用host网络模式**
```bash
# 启动容器时使用host网络
docker run --network host your_image
```

#### **方法B: 映射代理端口**
```bash
# 启动容器时映射端口
docker run -p 7890:7890 your_image
```

#### **方法C: 修改宿主机代理监听地址**
在宿主机上将代理配置为监听所有接口：
```bash
# 将代理从 127.0.0.1:7890 改为 0.0.0.0:7890
```

### **步骤3: 在Docker容器中使用**

```bash
# 自动配置并运行
python start.py

# 或使用智能启动脚本
./start_with_network_config.sh

# 手动控制
python proxy_control.py enable --force
python start.py
```

## 🔧 **故障排除**

### **问题1: 代理端口不可达**

**症状**: `❌ 代理端口不可达 (错误码: 11)`

**原因**: Docker容器无法访问宿主机的7890端口

**解决方案**:

1. **检查宿主机代理绑定地址**:
   ```bash
   # 在宿主机查看端口监听状态
   netstat -tlnp | grep 7890
   
   # 如果只监听127.0.0.1，需要改为0.0.0.0
   ```

2. **使用host网络模式**:
   ```bash
   docker run --network host your_image
   ```

3. **端口映射**:
   ```bash
   docker run -p 7890:7890 your_image
   ```

### **问题2: 代理可达但功能异常**

**症状**: 端口可达但代理功能测试失败

**解决方案**:
1. 检查代理服务器配置
2. 确认代理支持HTTP/HTTPS
3. 测试代理服务器本身的网络连接

### **问题3: 仍有地区限制错误**

**症状**: 使用代理后仍然收到地区限制错误

**可能原因**:
- 代理服务器也在受限地区
- 代理配置不正确
- AI服务检测到代理使用

**解决方案**:
1. 验证代理服务器的实际IP地址
2. 尝试不同的代理配置
3. 联系网络管理员确认代理设置

## 📊 **当前状态检查**

运行诊断命令：
```bash
# 完整网络诊断
python env_config.py

# 代理状态检查
python proxy_control.py status

# API访问测试
python proxy_control.py api
```

## 💡 **最佳实践建议**

### **1. 推荐配置**
```bash
# 在容器启动前设置环境变量
export USE_PROXY=True
export AUTO_PROXY=True

# 或修改 env_config.py 中的默认值
```

### **2. 监控和日志**
```bash
# 启用详细日志
export VERBOSE_NETWORK=True

# 定期检查网络状态
watch -n 30 "python proxy_control.py status"
```

### **3. 备用方案**
```bash
# 准备多个配置
export PROXY_FALLBACK_HOST=172.17.0.1
export PROXY_FALLBACK_PORT=8080

# 或使用其他AI服务
export MODEL=anthropic/claude-3-5-haiku-latest
```

## 📝 **总结**

### ✅ **已解决的问题**:
1. Docker容器内正确访问宿主机代理的地址映射
2. 智能代理检测和自动降级机制
3. 完整的代理控制和测试工具
4. 详细的故障排除指南

### ⚠️ **可能需要的额外配置**:
1. 宿主机代理服务绑定地址调整
2. Docker网络模式优化
3. 防火墙/安全组配置

### 🎯 **关键要点**:
- **宿主机地址**: `127.0.0.1:7890` → **Docker容器内地址**: `172.17.0.1:7890`
- 使用智能检测避免硬编码配置
- 提供完整的故障排除和控制工具
- 支持手动强制启用以应对特殊情况

现在您拥有了完整的Docker环境宿主机代理配置解决方案！ 
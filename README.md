# Gemini Native

![Version](https://img.shields.io/badge/version-v1.0--preview.8-2563eb)
![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078d4)
![Status](https://img.shields.io/badge/status-engineering%20preview-f59e0b)

Gemini Native 是面向 Windows 的动捕与机器人动作实验程序，支持动捕接入、动作预览、Unitree G1 仿真，以及实验性真机控制。

## 已实现功能

- 实时接收并预览动捕动作。
- 在仿真环境中查看机器人动作效果。
- 通过本机控制台完成设备设置、运行检查、启动、停止和日志查看。
- 支持 Unitree G1 官方 SDK；使用时无需手动选择 4010/5010 电机版本。
- 提供授权检查和真机操作前的安全确认。

## 使用说明

完整的获授权发布包面向 Windows 10/11 x64，可在目标电脑上运行，无需用户另行安装 WSL、Ubuntu、Conda 或系统 Python。

本仓库是**公开源码**，不包含运行时、模型、机器人资产、授权验证材料或客户授权；仅克隆仓库不能直接运行。请通过获授权渠道获取完整发布包和有效许可证。

首次使用建议先完成动捕连接与仿真检查，再考虑真机测试。参见 [首次运行说明](README_FIRST.txt)。

> [!WARNING]
> 真机控制仍属实验功能。测试时必须使用已受力的安全支撑架，清空机器人周围区域，并由操作员全程掌握实体遥控器与物理急停。网页停止按钮不能替代物理急停。真机操作请遵循 [快速指南](README_REAL_ROBOT.md) 和 [手柄说明](README_G1_REMOTE.md)。

## 发布与授权

客户许可证、签发密钥和现场设备信息不随源码发布，也不得上传到仓库或 Release。第三方许可文本见 [GMR 许可](GMR_LICENSE.txt)和 [Unitree SDK2 Python 许可](UNITREE_SDK2_PYTHON_LICENSE.txt)。

当前版本为工程预览版；真机功能需在具体设备与安全条件下另行验收。

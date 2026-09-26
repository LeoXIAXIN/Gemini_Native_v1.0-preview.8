(() => {
  "use strict";

  const API = {
    health: "/api/health",
    license: "/api/license?refresh=1",
    config: "/api/config",
    status: "/api/status",
    preflight: "/api/preflight",
    start: "/api/start",
    stop: "/api/stop",
    forceStop: "/api/force-stop",
    safetyEstop: "/api/safety/estop",
    safetyReset: "/api/safety/reset",
    cleanup: "/api/cleanup",
    shutdown: "/api/shutdown",
  };

  const POLL_INTERVAL_MS = 500;
  // Native preflight can spend 45s + 30s + 10s in runtime/SDK probes,
  // plus license verification (four 4s time sources). Start repeats preflight.
  // Keep status polling responsive while allowing these operations to finish.
  const REQUEST_TIMEOUT_MS = Object.freeze({
    [API.license]: 30000,
    [API.preflight]: 120000,
    [API.start]: 120000,
  });
  const MAX_LOG_LINES = 2000;
  const FORCE_STOP_DELAY_MS = 8000;

  const DEFAULT_CONFIG = Object.freeze({
    backend: "humanoid_gpt",
    execution_mode: "simulation_only",
    debug_mode: true,
    safety_strategy_enabled: false,
    server_ip: "192.168.2.100",
    unitree_network_interface: "192.168.123.100",
    unitree_robot_ip: "192.168.123.164",
    skeleton_id: 0,
    human_height: 1.75,
    publish_fps: 50,
    transition_seconds: 1.5,
    device: "auto",
  });

  const BACKENDS = Object.freeze({
    humanoid_gpt: {
      label: "实时生成式遥操作",
      shortLabel: "实时生成式遥操作",
      description: "动作控制",
      dof: 29,
    },
    gmr_preview: {
      label: "仅 GMR 预览",
      shortLabel: "仅 GMR 预览",
      description: "仅查看动作重定向结果",
      dof: 29,
    },
  });

  const EXECUTION_MODES = Object.freeze({
    simulation_only: {
      label: "仅仿真",
      tone: "safe",
      noticeTitle: "当前：仅仿真",
      noticeText: "不会加载 Unitree 真机接口，也不会发送任何电机指令。",
      startAllowed: true,
    },
    real_only: {
      label: "仅真机",
      tone: "experimental",
      noticeTitle: "实验性真机直通",
      noticeText: "不启动仿真；按实时生成式遥操作链路连接 G1。网页停止不是急停，必须保留实体遥控器门控、安全架和物理急停。",
      startAllowed: true,
    },
    parallel: {
      label: "仿真+真机",
      tone: "experimental",
      noticeTitle: "实验性并行输出",
      noticeText: "同时启用仿真和真机输出；资源占用更高。网页停止不是急停，必须保留实体遥控器门控、安全架和物理急停。",
      startAllowed: true,
    },
    shadow: {
      label: "安全影子（待接入）",
      tone: "pending",
      noticeTitle: "安全影子模式尚未就绪",
      noticeText: "计划运行仿真并只读真机状态；安全网关接入前不会启动，也不会发送电机指令。",
      startAllowed: false,
    },
  });

  const PHASES = Object.freeze({
    stopped: {
      code: "IDLE",
      title: "控制流程尚未启动",
      message: "确认动捕参数后，可先运行环境检查。",
      tone: "neutral",
      step: -1,
      summary: "等待启动",
    },
    preflight: {
      code: "CHECKING",
      title: "正在检查运行环境",
      message: "正在检查运行依赖、设备和本地端口。",
      tone: "active",
      step: 0,
      summary: "第 1 / 3 步",
    },
    starting: {
      code: "STARTING",
      title: "正在启动所选控制链路",
      message: "运行环境已就绪，正在启动本地服务。",
      tone: "active",
      step: 0,
      summary: "正在启动",
    },
    waiting_mocap: {
      code: "WAITING_INPUT",
      title: "等待动捕 Skeleton 数据",
      message: "请保持自然站立，系统将在第一帧建立机器人原点。",
      tone: "active",
      step: 1,
      summary: "第 2 / 3 步",
    },
    starting_backend: {
      code: "LOADING_ENGINE",
      title: "正在启动 Motion Engine",
      message: "已收到有效动作，正在初始化所选输出链路。",
      tone: "active",
      step: 2,
      summary: "第 3 / 3 步",
    },
    running: {
      code: "RUNNING",
      title: "控制流程运行中",
      message: "动作数据与所选 Motion Engine 正在运行。",
      tone: "ok",
      step: 2,
      summary: "运行中",
    },
    stopping: {
      code: "STOPPING",
      title: "正在停止控制流程",
      message: "正在先停止真机控制器，再依次关闭仿真与输入进程。此操作不是急停。",
      tone: "warning",
      step: 2,
      summary: "正在停止",
    },
    error: {
      code: "ERROR",
      title: "控制流程启动或运行失败",
      message: "请根据错误提示处理后重新检查。",
      tone: "danger",
      step: 0,
      summary: "需要处理",
    },
  });

  const state = {
    config: { ...DEFAULT_CONFIG },
    savedConfig: { ...DEFAULT_CONFIG },
    configDirty: false,
    status: null,
    serviceOnline: false,
    activeAction: null,
    estopPending: false,
    safetyResetPending: false,
    softwareEstopLatched: false,
    pollTimer: null,
    polling: false,
    lastStep: -1,
    stopRequestedAt: null,
    backendApiValues: new Map(),
    supportedBackends: null,
    logs: [],
    seenLogIds: new Set(),
    suppressedLogIds: new Set(),
    logsPaused: false,
    manualAlert: null,
    confirmAction: null,
    consoleShutdown: false,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const elements = {};

  class ApiError extends Error {
    constructor(message, details = "", status = 0) {
      super(message);
      this.name = "ApiError";
      this.details = details;
      this.status = status;
    }
  }

  function cacheElements() {
    Object.assign(elements, {
      productLicenseState: $("#productLicenseState"),
      productLicenseStateText: $("#productLicenseStateText"),
      serviceState: $("#serviceState"),
      serviceStateText: $("#serviceStateText"),
      offlineStrip: $("#offlineStrip"),
      retryConnectionButton: $("#retryConnectionButton"),
      pipelineTitle: $("#pipelineTitle"),
      pipelineMessage: $("#pipelineMessage"),
      stateCode: $("#stateCode"),
      largeStatusIndicator: $("#largeStatusIndicator"),
      activeModeChip: $("#activeModeChip"),
      executionModeChip: $("#executionModeChip"),
      runDuration: $("#runDuration"),
      simulationDuration: $("#simulationDuration"),
      lastUpdate: $("#lastUpdate"),
      configForm: $("#configForm"),
      unsavedBadge: $("#unsavedBadge"),
      serverIp: $("#serverIp"),
      unitreeInterfaceSettings: $("#unitreeInterfaceSettings"),
      unitreeNetworkInterface: $("#unitreeNetworkInterface"),
      unitreeRobotIp: $("#unitreeRobotIp"),
      skeletonId: $("#skeletonId"),
      humanHeight: $("#humanHeight"),
      publishFps: $("#publishFps"),
      transitionSeconds: $("#transitionSeconds"),
      device: $("#device"),
      deviceHint: $("#deviceHint"),
      executionModeNotice: $("#executionModeNotice"),
      executionModeNoticeTitle: $("#executionModeNoticeTitle"),
      executionModeNoticeText: $("#executionModeNoticeText"),
      debugMode: $("#debugMode"),
      debugModeHint: $("#debugModeHint"),
      safetyStrategyEnabled: $("#safetyStrategyEnabled"),
      safetyStrategySetting: $("#safetyStrategySetting"),
      safetyStrategyHint: $("#safetyStrategyHint"),
      safetyStrategyToggleLabel: $("#safetyStrategyToggleLabel"),
      resetConfigButton: $("#resetConfigButton"),
      saveConfigButton: $("#saveConfigButton"),
      alertPanel: $("#alertPanel"),
      alertLabel: $("#alertLabel"),
      alertTitle: $("#alertTitle"),
      alertMessage: $("#alertMessage"),
      alertDetails: $("#alertDetails"),
      alertDetailsWrapper: $("#alertDetailsWrapper"),
      alertActionButton: $("#alertActionButton"),
      startupStepper: $("#startupStepper"),
      progressSummary: $("#progressSummary"),
      operatorPrompt: $("#operatorPrompt"),
      preflightResults: $("#preflightResults"),
      captureHealth: $("#captureHealth"),
      sourceFps: $("#sourceFps"),
      metricSkeletonId: $("#metricSkeletonId"),
      frameId: $("#frameId"),
      captureFreshness: $("#captureFreshness"),
      gmrHealth: $("#gmrHealth"),
      bridgeFps: $("#bridgeFps"),
      rootZ: $("#rootZ"),
      rejectedFrames: $("#rejectedFrames"),
      anchorState: $("#anchorState"),
      backendHealth: $("#backendHealth"),
      backendName: $("#backendName"),
      backendDescription: $("#backendDescription"),
      policyFps: $("#policyFps"),
      policyDevice: $("#policyDevice"),
      modelState: $("#modelState"),
      simulationHealth: $("#simulationHealth"),
      packetAge: $("#packetAge"),
      viewerState: $("#viewerState"),
      safetyStatusCard: $("#safetyStatusCard"),
      safetyHealth: $("#safetyHealth"),
      safetyGatewayState: $("#safetyGatewayState"),
      safetyReason: $("#safetyReason"),
      safetyReferenceHealth: $("#safetyReferenceHealth"),
      safetyReferenceAge: $("#safetyReferenceAge"),
      safetyTeleopWeight: $("#safetyTeleopWeight"),
      safetyTeleopProgress: $("#safetyTeleopProgress"),
      safetyEstopLatch: $("#safetyEstopLatch"),
      safetyFaultLatch: $("#safetyFaultLatch"),
      safetyStrategyState: $("#safetyStrategyState"),
      safetyHardGuards: $("#safetyHardGuards"),
      safetyLimited: $("#safetyLimited"),
      logsDetails: $("#logsDetails"),
      logSourceFilter: $("#logSourceFilter"),
      logLevelFilter: $("#logLevelFilter"),
      autoScrollToggle: $("#autoScrollToggle"),
      pauseLogsButton: $("#pauseLogsButton"),
      clearLogsButton: $("#clearLogsButton"),
      copyLogsButton: $("#copyLogsButton"),
      logConsole: $("#logConsole"),
      emptyLogs: $("#emptyLogs"),
      logLines: $("#logLines"),
      logCount: $("#logCount"),
      logSummary: $("#logSummary"),
      logStatusDot: $("#logStatusDot"),
      cleanupButton: $("#cleanupButton"),
      exitConsoleButton: $("#exitConsoleButton"),
      preflightButton: $("#preflightButton"),
      startButton: $("#startButton"),
      stopButton: $("#stopButton"),
      forceStopButton: $("#forceStopButton"),
      emergencyStopButton: $("#emergencyStopButton"),
      safetyResetButton: $("#safetyResetButton"),
      dockSafetyTitle: $("#dockSafetyTitle"),
      dockSafetyText: $("#dockSafetyText"),
      confirmDialog: $("#confirmDialog"),
      confirmTitle: $("#confirmTitle"),
      confirmMessage: $("#confirmMessage"),
      confirmActionButton: $("#confirmActionButton"),
      toastRegion: $("#toastRegion"),
      liveStatus: $("#liveStatus"),
    });
  }

  function normalizeBackend(value) {
    const raw = String(value || DEFAULT_CONFIG.backend).trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (raw === "humanoidgpt" || raw === "hgpt") return "humanoid_gpt";
    if (raw === "preview" || raw === "gmr") return "gmr_preview";
    if (raw === "twist") return "humanoid_gpt";
    return BACKENDS[raw] ? raw : DEFAULT_CONFIG.backend;
  }

  function publicBackend(value) {
    return normalizeBackend(value);
  }

  function publicExecutionMode(value) {
    const mode = String(value || DEFAULT_CONFIG.execution_mode);
    if (mode === "shadow") return "real_only";
    return ["simulation_only", "real_only", "parallel"].includes(mode)
      ? mode
      : DEFAULT_CONFIG.execution_mode;
  }

  function normalizePhase(value, running = false) {
    const raw = String(value || (running ? "running" : "stopped")).trim().toLowerCase();
    const aliases = {
      idle: "stopped",
      ready: "stopped",
      stopped: "stopped",
      checking: "preflight",
      checking_environment: "preflight",
      waiting: "waiting_mocap",
      waiting_for_data: "waiting_mocap",
      waiting_for_udp_frame: "waiting_mocap",
      waiting_for_local_frame: "waiting_mocap",
      local_frame_received: "waiting_mocap",
      retargeting_first_frame: "waiting_mocap",
      loading: "starting_backend",
      loading_policy: "starting_backend",
      failed: "error",
      stopping: "stopping",
    };
    const normalized = aliases[raw] || raw;
    return PHASES[normalized] ? normalized : running ? "running" : "stopped";
  }

  function normalizeLevel(value) {
    const level = String(value || "info").toLowerCase();
    if (level.includes("err") || level === "fatal") return "error";
    if (level.includes("warn")) return "warning";
    return "info";
  }

  function normalizeLogSource(value) {
    const source = String(value || "system").toLowerCase();
    if (/chingmu|mocap|sender|sdk|vrpn/.test(source)) return "chingmu";
    if (/gmr|bridge|retarget/.test(source)) return "gmr";
    if (/twist|humanoid|gpt|policy|backend/.test(source)) return "backend";
    if (/mujoco|simulation|simulator|viewer/.test(source)) return "mujoco";
    return "system";
  }

  function sourceLabel(value) {
    return {
      system: "SYSTEM",
      chingmu: "INPUT",
      gmr: "MOTION",
      backend: "ENGINE",
      mujoco: "OUTPUT",
    }[value] || "SYSTEM";
  }

  function publicText(value) {
    return String(value ?? "")
      .replace(
        /[A-Za-z0-9_.-]*(?:ChingMu|MCAvatar|CMVrpn|VRPN|Humanoid[\s_-]*GPT|HGPT|TWIST|GMR|MuJoCo|ONNX|TensorRT|Redis|policy)[A-Za-z0-9_.-]*/gi,
        (token) => {
          const lowered = token.toLowerCase();
          const aliases = new Set();
          if (/chingmu|mcavatar|cmvrpn|vrpn/.test(lowered)) aliases.add("动捕");
          if (/humanoid[\s_-]*gpt|hgpt/.test(lowered)) aliases.add("实时生成式遥操作");
          if (/twist/.test(lowered)) aliases.add("内部组件");
          if (/gmr/.test(lowered)) aliases.add("动作转换");
          if (/mujoco/.test(lowered)) aliases.add("仿真");
          if (/onnx/.test(lowered)) aliases.add("模型运行时");
          if (/tensorrt/.test(lowered)) aliases.add("加速运行时");
          if (/redis/.test(lowered)) aliases.add("本地数据通道");
          if (/policy/.test(lowered)) aliases.add("engine");
          const publicAliases = Array.from(aliases);
          return publicAliases.length === 1 ? publicAliases[0] : "内部组件";
        }
      )
      .replace(/Humanoid[\s_-]*GPT/gi, "实时生成式遥操作")
      .replace(/\bHGPT\b/gi, "实时生成式遥操作")
      .replace(/\bTWIST\b/gi, "内部组件")
      .replace(/\bGMR\b/gi, "动作转换")
      .replace(/\bMuJoCo\b/gi, "仿真")
      .replace(/\bONNX\b/gi, "模型运行时")
      .replace(/\bTensorRT\b/gi, "加速运行时")
      .replace(/\bRedis\b/gi, "本地数据通道")
      .replace(/\bpolicy\b/gi, "engine")
      .replace(/青瞳/g, "动捕")
      .replace(/策略模型/g, "控制模型")
      .replace(/策略后端/g, "动作引擎")
      .replace(/策略/g, "控制");
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizeBoolean(value, fallback = false) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["1", "true", "yes", "on", "enabled"].includes(normalized)) return true;
      if (["0", "false", "no", "off", "disabled"].includes(normalized)) return false;
    }
    return Boolean(fallback);
  }

  function optionalBoolean(value) {
    if (typeof value === "boolean") return value;
    if (value === 0 || value === 1) return value === 1;
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["1", "true", "yes", "on", "enabled"].includes(normalized)) return true;
      if (["0", "false", "no", "off", "disabled"].includes(normalized)) return false;
    }
    return null;
  }

  function formatNumber(value, digits = 1) {
    const number = finiteNumber(value);
    return number === null ? "—" : number.toFixed(digits);
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(finiteNumber(seconds) || 0));
    const hours = String(Math.floor(total / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
    const remaining = String(total % 60).padStart(2, "0");
    return `${hours}:${minutes}:${remaining}`;
  }

  function formatLogTime(value) {
    if (!value) return new Date().toLocaleTimeString("zh-CN", { hour12: false });
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleTimeString("zh-CN", { hour12: false });
    }
    const text = String(value);
    const match = text.match(/\d{1,2}:\d{2}:\d{2}/);
    return match ? match[0].padStart(8, "0") : text.slice(0, 8);
  }

  function setHealthBadge(element, text, tone = "neutral") {
    element.textContent = text;
    element.dataset.tone = tone;
  }

  async function fetchJson(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS[url] ?? 6000) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    const method = String(options.method || "GET").toUpperCase();
    const isJsonMutation = ["POST", "PUT", "PATCH", "DELETE"].includes(method);
    const request = {
      ...options,
      method,
      body: isJsonMutation && options.body === undefined ? "{}" : options.body,
      headers: {
        Accept: "application/json",
        ...(isJsonMutation ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
      signal: controller.signal,
      cache: "no-store",
    };

    try {
      const response = await fetch(url, request);
      const text = await response.text();
      let payload = {};
      if (text) {
        try {
          payload = JSON.parse(text);
        } catch {
          payload = { message: text };
        }
      }
      if (!response.ok) {
        const message = payload.error || payload.message || `请求失败（HTTP ${response.status}）`;
        throw new ApiError(String(message), text, response.status);
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new ApiError("控制服务响应超时", `${url} 未在 ${timeoutMs} ms 内响应`);
      }
      if (error instanceof ApiError) throw error;
      throw new ApiError("无法连接本地控制服务", String(error.message || error));
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function extractStatus(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (payload.status && typeof payload.status === "object") return payload.status;
    return payload.phase || Object.hasOwn(payload, "running") ? payload : null;
  }

  function setServiceOnline(online) {
    if (state.consoleShutdown) return;
    state.serviceOnline = online;
    elements.serviceState.dataset.tone = online ? "ok" : "danger";
    elements.serviceStateText.textContent = online ? "本地控制服务已连接" : "控制服务未连接";
    elements.offlineStrip.hidden = online;
    updateActionButtons();
  }

  function setProductLicenseStatus(payload = null) {
    const allowed = payload?.allowed === true;
    const checked = payload && payload.code && payload.code !== "not_checked";
    elements.productLicenseState.dataset.tone = allowed
      ? "ok"
      : checked ? "danger" : "neutral";
    if (allowed && payload.license_type === "online_trial" && payload.valid_until) {
      const expiry = new Date(payload.valid_until);
      elements.productLicenseStateText.textContent = Number.isNaN(expiry.getTime())
        ? "授权：期限有效"
        : `授权：有效至 ${expiry.toLocaleDateString("zh-CN")}`;
    } else if (allowed && payload.license_type === "dongle") {
      elements.productLicenseStateText.textContent = "授权：加密狗有效";
    } else if (allowed && payload.license_type === "cloud") {
      elements.productLicenseStateText.textContent = "授权：云端有效";
    } else {
      elements.productLicenseStateText.textContent = checked
        ? "授权：不可用"
        : "授权：正在验证";
    }
    elements.productLicenseState.title = payload?.message
      || "授权只决定能否开始新任务，不替代 ARM G1 真机安全授权";
  }

  async function refreshProductLicense() {
    try {
      setProductLicenseStatus(await fetchJson(API.license));
    } catch (error) {
      setProductLicenseStatus({
        allowed: false,
        code: "license_check_unavailable",
        message: error.message || "无法完成联网授权检查",
      });
    }
  }

  function selectedBackend() {
    return publicBackend($("input[name='mode']:checked", elements.configForm)?.value);
  }

  function selectedExecutionMode() {
    return publicExecutionMode(
      $("input[name='execution_mode']:checked", elements.configForm)?.value
        || DEFAULT_CONFIG.execution_mode
    );
  }

  function executionUsesUnitree(mode = selectedExecutionMode()) {
    return ["real_only", "parallel", "shadow"].includes(mode);
  }

  function collectConfig() {
    updateSafetyStrategyControl();
    const normalized = selectedBackend();
    return {
      backend: state.backendApiValues.get(normalized) || normalized,
      execution_mode: selectedExecutionMode(),
      debug_mode: elements.debugMode.checked,
      safety_strategy_enabled: elements.safetyStrategyEnabled.checked,
      server_ip: elements.serverIp.value.trim(),
      unitree_network_interface: elements.unitreeNetworkInterface.value.trim(),
      unitree_robot_ip: elements.unitreeRobotIp.value.trim(),
      skeleton_id: Number(elements.skeletonId.value),
      human_height: Number(elements.humanHeight.value),
      publish_fps: Number(elements.publishFps.value),
      transition_seconds: Number(elements.transitionSeconds.value),
      device: elements.device.value,
    };
  }

  function canonicalConfig(config) {
    const source = config && typeof config === "object" ? config : {};
    const backend = publicBackend(source.backend ?? source.mode ?? DEFAULT_CONFIG.backend);
    const rawExecutionMode = String(source.execution_mode ?? DEFAULT_CONFIG.execution_mode);
    const executionMode = publicExecutionMode(rawExecutionMode);
    const debugMode = rawExecutionMode === "shadow"
      ? true
      : normalizeBoolean(source.debug_mode, DEFAULT_CONFIG.debug_mode);
    const safetyStrategySupported = backend === "humanoid_gpt"
      && ["simulation_only", "parallel"].includes(executionMode);
    const requestedSafety = normalizeBoolean(
      source.safety_strategy_enabled,
      DEFAULT_CONFIG.safety_strategy_enabled
    );
    return {
      backend,
      execution_mode: executionMode,
      debug_mode: debugMode,
      safety_strategy_enabled: safetyStrategySupported ? requestedSafety : false,
      server_ip: String(source.server_ip ?? DEFAULT_CONFIG.server_ip),
      unitree_network_interface: String(
        source.unitree_network_interface ?? DEFAULT_CONFIG.unitree_network_interface
      ).trim(),
      unitree_robot_ip: String(source.unitree_robot_ip ?? DEFAULT_CONFIG.unitree_robot_ip).trim(),
      skeleton_id: Number(source.skeleton_id ?? DEFAULT_CONFIG.skeleton_id),
      human_height: Number(source.human_height ?? DEFAULT_CONFIG.human_height),
      publish_fps: Number(source.publish_fps ?? source.frequency ?? DEFAULT_CONFIG.publish_fps),
      transition_seconds: Number(source.transition_seconds ?? DEFAULT_CONFIG.transition_seconds),
      device: String(source.device ?? DEFAULT_CONFIG.device),
    };
  }

  function applyConfig(config, markSaved = true) {
    const next = canonicalConfig(config);
    state.config = next;
    if (markSaved) state.savedConfig = { ...next };

    const modeInput = $(`input[name="mode"][value="${next.backend}"]`, elements.configForm);
    if (modeInput) modeInput.checked = true;
    const executionInput = $(`input[name="execution_mode"][value="${next.execution_mode}"]`, elements.configForm);
    if (executionInput) executionInput.checked = true;
    elements.debugMode.checked = next.debug_mode;
    elements.safetyStrategyEnabled.checked = next.safety_strategy_enabled;
    elements.serverIp.value = next.server_ip;
    elements.unitreeNetworkInterface.value = next.unitree_network_interface;
    elements.unitreeRobotIp.value = next.unitree_robot_ip;
    elements.skeletonId.value = String(next.skeleton_id);
    elements.humanHeight.value = String(next.human_height);
    elements.publishFps.value = String(next.publish_fps);
    elements.transitionSeconds.value = String(next.transition_seconds);
    if ($(`option[value="${CSS.escape(next.device)}"]`, elements.device)) {
      elements.device.value = next.device;
    }

    setDirty(false);
    updateModePresentation();
  }

  function optionValue(item) {
    if (typeof item === "string" || typeof item === "number") return String(item);
    return String(item?.value ?? item?.id ?? item?.name ?? "");
  }

  function optionLabel(item, fallback) {
    if (!item || typeof item !== "object") return fallback;
    return String(item.label ?? item.title ?? item.name ?? fallback);
  }

  function applyOptions(options) {
    if (!options || typeof options !== "object") return;

    if (Array.isArray(options.backends) && options.backends.length) {
      state.backendApiValues.clear();
      state.supportedBackends = new Set();
      options.backends.forEach((item) => {
        const apiValue = optionValue(item);
        const normalized = normalizeBackend(apiValue);
        state.backendApiValues.set(normalized, apiValue);
        state.supportedBackends.add(normalized);
      });
    }

    const populateSelect = (select, items, currentValue) => {
      if (!Array.isArray(items) || !items.length) return;
      const fragment = document.createDocumentFragment();
      items.forEach((item) => {
        const value = optionValue(item);
        if (!value) return;
        const option = document.createElement("option");
        option.value = value;
        option.textContent = optionLabel(item, value.toUpperCase());
        fragment.append(option);
      });
      select.replaceChildren(fragment);
      if ($(`option[value="${CSS.escape(String(currentValue))}"]`, select)) {
        select.value = String(currentValue);
      }
    };

    populateSelect(elements.device, options.devices, state.config.device);
  }

  function clearFieldErrors() {
    $$(".field-error", elements.configForm).forEach((element) => {
      element.textContent = "";
    });
    $$("input[aria-invalid='true'], select[aria-invalid='true']", elements.configForm).forEach((element) => {
      element.removeAttribute("aria-invalid");
    });
  }

  function setFieldError(input, message) {
    input.setAttribute("aria-invalid", "true");
    const error = $(`#${CSS.escape(input.id)}Error`);
    if (error) error.textContent = message;
  }

  function isValidIpv4(value) {
    const parts = String(value).split(".");
    return parts.length === 4 && parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255);
  }

  function validateConfig(focusFirst = true) {
    clearFieldErrors();
    const config = collectConfig();
    const invalid = [];

    if (!isValidIpv4(config.server_ip)) {
      setFieldError(elements.serverIp, "请输入有效的 IPv4 地址，例如 192.168.2.100");
      invalid.push(elements.serverIp);
    }
    if (!Number.isInteger(config.skeleton_id) || config.skeleton_id < 0 || config.skeleton_id > 255) {
      setFieldError(elements.skeletonId, "Skeleton ID 必须是 0–255 的整数");
      invalid.push(elements.skeletonId);
    }
    if (!Number.isFinite(config.human_height) || config.human_height < 1.2 || config.human_height > 2.2) {
      setFieldError(elements.humanHeight, "人体身高范围为 1.20–2.20 m");
      invalid.push(elements.humanHeight);
    }
    if (!Number.isFinite(config.publish_fps) || config.publish_fps < 20 || config.publish_fps > 60) {
      setFieldError(elements.publishFps, "目标频率范围为 20–60 Hz");
      invalid.push(elements.publishFps);
    }
    if (!Number.isFinite(config.transition_seconds) || config.transition_seconds < 0.5 || config.transition_seconds > 5) {
      setFieldError(elements.transitionSeconds, "平滑过渡范围为 0.5–5 秒");
      invalid.push(elements.transitionSeconds);
    }
    if (config.safety_strategy_enabled && !(
      normalizeBackend(config.backend) === "humanoid_gpt"
      && ["simulation_only", "parallel"].includes(config.execution_mode)
    )) {
      setFieldError(
        elements.safetyStrategyEnabled,
        "实验性二级安全策略目前只支持实时生成式遥操作的仿真分支"
      );
      invalid.push(elements.safetyStrategyEnabled);
    }
    if (executionUsesUnitree(config.execution_mode)) {
      if (!isValidIpv4(config.unitree_network_interface)) {
        setFieldError(elements.unitreeNetworkInterface, "请输入连接 G1 的 Windows 网卡 IPv4，例如 192.168.123.100");
        invalid.push(elements.unitreeNetworkInterface);
      } else if (config.unitree_network_interface.startsWith("127.")) {
        setFieldError(elements.unitreeNetworkInterface, "本机回环地址不能用于连接 Unitree G1");
        invalid.push(elements.unitreeNetworkInterface);
      }
      if (!isValidIpv4(config.unitree_robot_ip)) {
        setFieldError(elements.unitreeRobotIp, "请输入有效的 G1 IPv4 地址，例如 192.168.123.164");
        invalid.push(elements.unitreeRobotIp);
      }
    }

    if (invalid.length && focusFirst) invalid[0].focus();
    return invalid.length === 0;
  }

  function setDirty(dirty) {
    state.configDirty = Boolean(dirty);
    elements.unsavedBadge.hidden = !dirty;
    elements.saveConfigButton.textContent = dirty ? "保存设置" : "设置已保存";
    updateActionButtons();
  }

  function updateSafetyStrategyControl(locked = isConfigLocked()) {
    const executionMode = selectedExecutionMode();
    const parallel = executionMode === "parallel";
    const supported = selectedBackend() === "humanoid_gpt"
      && ["simulation_only", "parallel"].includes(executionMode);

    if (!supported) elements.safetyStrategyEnabled.checked = false;
    elements.safetyStrategyEnabled.disabled = locked || !supported;
    elements.safetyStrategySetting.dataset.eligible = String(supported && !locked);
    elements.safetyStrategySetting.dataset.mode = elements.safetyStrategyEnabled.checked
      ? "experimental"
      : "off";
    elements.safetyStrategyToggleLabel.textContent = elements.safetyStrategyEnabled.checked
      ? "已启用（实验）"
      : supported ? "关闭（默认）" : "不适用 · 保持关闭";

    if (locked) {
      elements.safetyStrategyHint.textContent = "流程运行中不可修改；请先停止，保存设置后重新启动。";
    } else if (!supported) {
      elements.safetyStrategyHint.textContent = "当前分支不支持实验性二级安全策略，已保持关闭；真机模式走上游原生部署链路。";
    } else if (elements.safetyStrategyEnabled.checked) {
      elements.safetyStrategyHint.textContent = parallel
        ? "已启用，但只保护并行模式中的仿真分支；真机仍为原生直通。"
        : "仅对实时生成式遥操作的仿真分支启用实验性软件安全状态机。";
    } else {
      elements.safetyStrategyHint.textContent = parallel
        ? "默认关闭：仿真与真机均保持原生链路。"
        : "默认关闭：保持加入实验性策略前的原生仿真链路。";
    }
  }

  function updateModePresentation() {
    const backend = selectedBackend();
    if (backend === "gmr_preview" && selectedExecutionMode() !== "simulation_only") {
      const simulationOnly = document.querySelector('input[name="execution_mode"][value="simulation_only"]');
      if (simulationOnly) simulationOnly.checked = true;
    }
    const info = BACKENDS[backend];
    const executionMode = selectedExecutionMode();
    const executionInfo = EXECUTION_MODES[executionMode];
    elements.activeModeChip.textContent = info.label;
    elements.executionModeChip.textContent = executionInfo.label;
    elements.backendName.textContent = info.shortLabel;
    elements.backendDescription.textContent = info.description;
    elements.executionModeNotice.dataset.tone = executionInfo.tone;
    elements.executionModeNoticeTitle.textContent = executionInfo.noticeTitle;
    elements.executionModeNoticeText.textContent = executionInfo.noticeText;
    const unitreeMode = executionUsesUnitree(executionMode);
    elements.unitreeInterfaceSettings.dataset.active = String(unitreeMode);
    elements.unitreeInterfaceSettings.hidden = !unitreeMode;
    elements.unitreeInterfaceSettings.setAttribute("aria-hidden", String(!unitreeMode));
    elements.debugModeHint.textContent = unitreeMode
      ? "真机调试模式只订阅 LowState，不发布 LowCmd，用于检查网卡、DDS 与状态链路。"
      : "仿真中输出诊断信息；修改配置后必须停止并重新启动流程。";

    const preview = backend === "gmr_preview";
    elements.device.disabled = preview || isConfigLocked();
    elements.deviceHint.textContent = "";
    elements.startButton.querySelector(".button-label").textContent = !executionInfo.startAllowed
      ? "影子模式待接入"
      : preview
        ? "启动预览"
        : executionMode === "real_only"
          ? "启动实验性真机直通"
          : executionMode === "parallel"
            ? "启动仿真 + 真机"
            : "启动仿真";
    elements.stopButton.querySelector(".button-label").textContent = unitreeMode
      ? "停止控制（非急停）"
      : preview ? "停止预览" : "停止仿真";
    elements.policyDevice.textContent = preview ? "不适用" : elements.device.value.toUpperCase();
    updateConfigLock();
    updateActionButtons();
  }

  function isConfigLocked() {
    const phase = normalizePhase(state.status?.phase, state.status?.running);
    return Boolean(state.status?.running) || !["stopped", "error"].includes(phase) || Boolean(state.activeAction);
  }

  function updateConfigLock() {
    const locked = isConfigLocked();
    $$("input, select", elements.configForm).forEach((control) => {
      const mode = control.name === "mode" ? normalizeBackend(control.value) : null;
      const unsupported = mode && state.supportedBackends && !state.supportedBackends.has(mode);
      const previewRealOutput = control.name === "execution_mode"
        && selectedBackend() === "gmr_preview"
        && ["real_only", "parallel", "shadow"].includes(control.value);
      const unitreeControl = control === elements.unitreeNetworkInterface
        || control === elements.unitreeRobotIp;
      const notApplicable = (control === elements.device && selectedBackend() === "gmr_preview")
        || previewRealOutput
        || (unitreeControl && !executionUsesUnitree());
      control.disabled = locked || unsupported || notApplicable;
    });
    updateSafetyStrategyControl(locked);
    elements.resetConfigButton.disabled = locked;
    elements.saveConfigButton.disabled = locked || !state.configDirty;
  }

  async function persistConfig(showSuccess = true) {
    if (!validateConfig()) throw new ApiError("请先修正配置中的错误");
    const config = collectConfig();
    const payload = await fetchJson(API.config, {
      method: "PUT",
      body: JSON.stringify(config),
    });
    const saved = payload.config || config;
    applyConfig(saved, true);
    if (payload.options) applyOptions(payload.options);
    if (showSuccess) showToast("设置已保存", "success");
    return saved;
  }

  function beginAction(name, button) {
    if (state.activeAction) return false;
    state.activeAction = name;
    if (button) button.classList.add("is-loading");
    updateConfigLock();
    updateActionButtons();
    return true;
  }

  function endAction(button) {
    if (button) button.classList.remove("is-loading");
    state.activeAction = null;
    updateConfigLock();
    updateActionButtons();
  }

  function updateActionButtons() {
    const phase = normalizePhase(state.status?.phase, state.status?.running);
    const running = Boolean(state.status?.running) || ["starting", "waiting_mocap", "starting_backend", "running", "stopping"].includes(phase);
    const stopping = phase === "stopping";
    const busy = Boolean(state.activeAction);
    const canPrepare = state.serviceOnline && !running && ["stopped", "error"].includes(phase);
    const executionInfo = EXECUTION_MODES[selectedExecutionMode()];
    const pipelineActive = state.status?.active === undefined
      ? running
      : Boolean(state.status.active);
    const activeWorkflow = state.serviceOnline && pipelineActive;
    const activeBackend = normalizeBackend(
      state.status?.backend || state.config.backend || selectedBackend()
    );
    const activeExecutionMode = state.config.execution_mode || selectedExecutionMode();
    const activeSafety = state.status?.safety;
    const reportedSafetyEnabled = optionalBoolean(
      activeSafety?.strategy_enabled
      ?? activeSafety?.safety_strategy_enabled
      ?? activeSafety?.strategy?.enabled
    );
    const reportedGatewayState = String(
      activeSafety?.gateway_state ?? activeSafety?.state ?? ""
    ).trim().toUpperCase();
    const legacyDirectActive = reportedSafetyEnabled === false
      || normalizeBoolean(
        activeSafety?.legacy_direct
        ?? activeSafety?.bypass_active
        ?? activeSafety?.bypassed,
        false
      )
      || ["LEGACY_DIRECT", "BYPASS_TRACKING"].includes(reportedGatewayState)
      || (reportedSafetyEnabled === null && state.config.safety_strategy_enabled === false);
    const webResetAllowed = activeWorkflow
      && !stopping
      && activeBackend === "humanoid_gpt"
      && activeExecutionMode === "simulation_only"
      && !legacyDirectActive;
    const softwareEstopAvailable = activeWorkflow
      && activeBackend === "humanoid_gpt"
      && activeExecutionMode === "simulation_only"
      && !legacyDirectActive;

    elements.startButton.hidden = running;
    elements.stopButton.hidden = !running;
    elements.cleanupButton.hidden = running;
    elements.preflightButton.hidden = running;
    elements.startButton.disabled = busy || !canPrepare || !executionInfo.startAllowed;
    elements.startButton.title = executionInfo.startAllowed
      ? ""
      : executionInfo.noticeText;
    elements.preflightButton.disabled = busy || !canPrepare;
    elements.cleanupButton.disabled = busy || !state.serviceOnline || !["stopped", "error"].includes(phase);
    elements.exitConsoleButton.disabled = busy || !state.serviceOnline || running || !["stopped", "error"].includes(phase);
    elements.exitConsoleButton.title = running ? "请先停止当前流程，再退出控制台" : "退出本地网页控制服务";
    elements.stopButton.disabled = busy || stopping;

    const showForce = stopping && state.stopRequestedAt && Date.now() - state.stopRequestedAt >= FORCE_STOP_DELAY_MS;
    elements.forceStopButton.hidden = !showForce;
    elements.forceStopButton.disabled = busy;

    // The E-stop deliberately bypasses the normal UI action lock.  It writes a
    // latch command and leaves the control process alive so damping can run.
    elements.emergencyStopButton.disabled = !softwareEstopAvailable || state.estopPending;
    elements.emergencyStopButton.classList.toggle("is-loading", state.estopPending);
    elements.emergencyStopButton.setAttribute(
      "aria-pressed",
      state.softwareEstopLatched ? "true" : "false"
    );
    elements.emergencyStopButton.title = softwareEstopAvailable
      ? "立即锁存仿真安全状态；不会停止进程，也不替代真机物理急停"
      : legacyDirectActive
        ? "实验性二级安全策略未启用；网页软件急停不可用，真机必须使用实体遥控器急停"
      : activeWorkflow
        ? "当前仅实时生成式遥操作纯仿真已接入软件安全状态机"
        : "仅在实时生成式遥操作纯仿真流程活动时可用";
    elements.safetyResetButton.hidden = !webResetAllowed;
    elements.safetyResetButton.disabled = state.estopPending || state.safetyResetPending;
    elements.safetyResetButton.classList.toggle("is-loading", state.safetyResetPending);
    elements.safetyResetButton.title = webResetAllowed
      ? "仅复位实时生成式遥操作纯仿真的软件安全锁"
      : legacyDirectActive
        ? "实验性二级安全策略未启用，因此没有可复位的网页锁存状态"
        : "当前分支没有可复位的网页安全锁";

    elements.saveConfigButton.disabled = busy || isConfigLocked() || !state.configDirty;
    elements.resetConfigButton.disabled = busy || isConfigLocked();
  }

  function renderPreflight(payload) {
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    elements.preflightResults.replaceChildren();
    if (!checks.length) {
      elements.preflightResults.hidden = true;
      return;
    }

    const fragment = document.createDocumentFragment();
    checks.forEach((check) => {
      const row = document.createElement("div");
      row.className = `check-result${check.ok ? "" : " is-failed"}`;

      const icon = document.createElement("span");
      icon.className = "check-result-icon";
      icon.textContent = check.ok ? "✓" : "!";

      const content = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = publicText(check.name || "检查项目");
      const message = document.createElement("span");
      message.textContent = publicText(check.message || (check.ok ? "正常" : "未通过"));
      content.append(title, message);
      row.append(icon, content);
      fragment.append(row);
    });
    elements.preflightResults.append(fragment);
    elements.preflightResults.hidden = false;
  }

  async function runPreflight() {
    if (state.configDirty) await persistConfig(false);
    else if (!validateConfig()) throw new ApiError("请先修正配置中的错误");

    state.manualAlert = null;
    renderAlert();
    const payload = await fetchJson(API.preflight, { method: "POST" });
    renderPreflight(payload);

    const failedChecks = Array.isArray(payload.checks)
      ? payload.checks.filter((item) => !item.ok && item.required !== false)
      : [];
    const errors = Array.isArray(payload.errors) ? payload.errors : [];
    if (payload.ok === false || failedChecks.length || errors.length) {
      const details = [...new Set([
        ...failedChecks.map((item) => publicText(`${item.name}: ${item.message}`)),
        ...errors.map((item) => publicText(item)),
      ])].join("\n");
      state.manualAlert = {
        tone: "danger",
        title: "环境检查未通过",
        message: "请先处理未通过的项目，再重新启动。",
        details,
        action: "preflight",
        actionLabel: "重新检查",
      };
      renderAlert();
      return false;
    }

    showToast("环境检查通过，可以启动", "success");
    return true;
  }

  async function handleSaveConfig(event) {
    event.preventDefault();
    if (!beginAction("save", elements.saveConfigButton)) return;
    try {
      await persistConfig(true);
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.saveConfigButton);
    }
  }

  async function handleStandalonePreflight() {
    if (!beginAction("preflight", elements.preflightButton)) return;
    try {
      await runPreflight();
      await pollStatus(true);
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.preflightButton);
    }
  }

  async function handleStart() {
    if (!beginAction("start", elements.startButton)) return;
    try {
      if (state.configDirty) await persistConfig(false);
      else if (!validateConfig()) throw new ApiError("请先修正配置中的错误");

      const passed = await runPreflight();
      if (!passed) return;
      const launchConfig = collectConfig();
      let realOutputAuthorization = null;
      if (executionUsesUnitree(launchConfig.execution_mode) && !launchConfig.debug_mode) {
        const acknowledgement = window.prompt(
          "即将允许向 Unitree G1 发布 LowCmd。网页停止不是急停。\n"
          + "请先在安全架上进入原厂阻尼/零力矩，再按 L2+R2 进入 Develop；"
          + "确认物理急停和遥控器均在手边后，输入 ARM G1："
        );
        if (acknowledgement !== "ARM G1") {
          showToast("未输入 ARM G1，已取消真机命令发送", "warning");
          return;
        }
        realOutputAuthorization = acknowledgement;
      }
      state.manualAlert = null;
      const payload = await fetchJson(API.start, {
        method: "POST",
        body: JSON.stringify(
          realOutputAuthorization === null
            ? {}
            : { authorization: realOutputAuthorization },
        ),
      });
      const status = extractStatus(payload);
      if (status) applyStatus(status);
      const executionMode = selectedExecutionMode();
      const message = selectedBackend() === "gmr_preview"
        ? "正在启动动作预览"
        : executionMode === "real_only"
          ? "正在启动实验性真机直通"
          : executionMode === "parallel"
            ? "正在启动仿真与真机并行输出"
            : "正在启动仿真";
      showToast(message, "success");
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.startButton);
      await pollStatus(true);
    }
  }

  async function handleStop() {
    if (!beginAction("stop", elements.stopButton)) return;
    state.stopRequestedAt = Date.now();
    try {
      const payload = await fetchJson(API.stop, { method: "POST" }, 15000);
      const status = extractStatus(payload);
      if (status) applyStatus(status);
      showToast("已发送停止流程请求；这不是急停");
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.stopButton);
      await pollStatus(true);
    }
  }

  async function handleEmergencyStop() {
    if (state.estopPending || elements.emergencyStopButton.disabled) return;
    state.estopPending = true;
    updateActionButtons();
    try {
      await fetchJson(API.safetyEstop, { method: "POST" }, 3000);
      state.softwareEstopLatched = true;
      showToast("仿真急停已锁存；流程仍在运行以维持安全控制", "danger");
      elements.liveStatus.textContent = "仿真急停已锁存。需要明确复位后才能恢复。";
    } catch (error) {
      handleActionError(error);
    } finally {
      state.estopPending = false;
      updateActionButtons();
    }
  }

  async function handleSafetyReset() {
    if (state.safetyResetPending || elements.safetyResetButton.hidden) return;
    const confirmed = await requestConfirmation({
      title: "复位仿真安全锁？",
      message: "仅限实时生成式遥操作纯仿真。请确认急停原因已排除、仿真画面处于安全状态。网页永远不能复位真机急停。",
      confirmLabel: "确认复位仿真安全锁",
    });
    if (!confirmed || state.estopPending) return;

    state.safetyResetPending = true;
    updateActionButtons();
    try {
      await fetchJson(API.safetyReset, {
        method: "POST",
        body: JSON.stringify({
          operator_confirmed: true,
          physical_estop_released: true,
        }),
      }, 3000);
      state.softwareEstopLatched = false;
      showToast("仿真安全锁已复位；控制器回到未使能状态", "success");
    } catch (error) {
      handleActionError(error);
    } finally {
      state.safetyResetPending = false;
      updateActionButtons();
    }
  }

  async function handleForceStop() {
    const confirmed = await requestConfirmation({
      title: "确认强制停止？",
      message: "仅在正常停止长时间无响应时使用。系统将终止本控制台启动的控制进程；这不是物理急停。",
      confirmLabel: "强制停止",
    });
    if (!confirmed || !beginAction("force-stop", elements.forceStopButton)) return;
    try {
      const payload = await fetchJson(API.forceStop, { method: "POST" }, 15000);
      const status = extractStatus(payload);
      if (status) applyStatus(status);
      showToast("控制进程已强制停止；这不是物理急停");
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.forceStopButton);
      await pollStatus(true);
    }
  }

  async function handleCleanup() {
    const confirmed = await requestConfirmation({
      title: "清理残留进程？",
      message: "仅清理本工程固定的输入、动作转换与控制进程，并释放本地端口和启动锁。",
      confirmLabel: "确认清理",
    });
    if (!confirmed || !beginAction("cleanup", elements.cleanupButton)) return;
    try {
      const payload = await fetchJson(API.cleanup, { method: "POST" }, 20000);
      const status = extractStatus(payload);
      if (status) applyStatus(status);
      state.manualAlert = null;
      showToast("残留进程与端口已清理", "success");
    } catch (error) {
      handleActionError(error);
    } finally {
      endAction(elements.cleanupButton);
      await pollStatus(true);
    }
  }

  async function handleExitConsole() {
    const phase = normalizePhase(state.status?.phase, state.status?.running);
    if (state.status?.running || !["stopped", "error"].includes(phase)) {
      showToast("请先停止当前流程，再退出控制台");
      return;
    }
    const confirmed = await requestConfirmation({
      title: "退出控制台？",
      message: "这会关闭本地网页控制服务，但不会删除已保存的设置。下次可再次运行启动程序。",
      confirmLabel: "退出控制台",
    });
    if (!confirmed || !beginAction("shutdown", elements.exitConsoleButton)) return;
    try {
      await fetchJson(API.shutdown, { method: "POST" }, 8000);
      markConsoleShutdown();
    } catch (error) {
      handleActionError(error);
      endAction(elements.exitConsoleButton);
    }
  }

  function markConsoleShutdown() {
    state.consoleShutdown = true;
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    elements.serviceState.dataset.tone = "neutral";
    elements.serviceStateText.textContent = "控制台已退出";
    elements.offlineStrip.hidden = true;
    elements.pipelineTitle.textContent = "控制台已安全退出";
    elements.pipelineMessage.textContent = "可以关闭此页面；下次使用时重新运行桌面启动程序。";
    elements.stateCode.textContent = "SHUTDOWN";
    elements.largeStatusIndicator.dataset.tone = "neutral";
    elements.liveStatus.textContent = "本地控制台已安全退出。";
    $$("button", document).forEach((button) => {
      if (!button.closest("dialog")) button.disabled = true;
    });
    showToast("控制台已退出，可以关闭此页面", "success");
  }

  function requestConfirmation({ title, message, confirmLabel }) {
    if (!elements.confirmDialog.showModal) {
      return Promise.resolve(window.confirm(`${title}\n\n${message}`));
    }
    elements.confirmTitle.textContent = title;
    elements.confirmMessage.textContent = message;
    elements.confirmActionButton.textContent = confirmLabel;
    elements.confirmDialog.showModal();
    return new Promise((resolve) => {
      const onClose = () => {
        elements.confirmDialog.removeEventListener("close", onClose);
        resolve(elements.confirmDialog.returnValue === "confirm");
      };
      elements.confirmDialog.addEventListener("close", onClose);
    });
  }

  function handleActionError(error) {
    const info = humanizeError(error.message || String(error), error.details || "");
    state.manualAlert = info;
    renderAlert();
    showToast(info.title, "danger");
  }

  function humanizeError(error, details = "") {
    const raw = `${error || ""}\n${details || ""}`;
    const base = {
      tone: "danger",
      title: "操作未完成",
      message: publicText(error || "请查看技术详情并重新检查。"),
      details: publicText(details || error || ""),
      action: "preflight",
      actionLabel: "重新检查",
    };

    if (/frame_bind|frame socket|local frame transport|unix:.*(?:active|占用)/i.test(raw)) {
      return { ...base, title: "本地帧通道无法启动", message: "可能存在上次异常退出后留下的接收进程或不安全的同名文件。", action: "cleanup", actionLabel: "清理残留" };
    }
    if (/address already in use|udp_bind|errno\s*98|eaddrinuse|15150.*占用/i.test(raw)) {
      return { ...base, title: "本地端口 15150 被占用", message: "可能存在上次异常退出后留下的接收进程。", action: "cleanup", actionLabel: "清理残留" };
    }
    if (/真机输出未安全解锁|real_output_enabled|unitree.*(?:locked|未解锁)/i.test(raw)) {
      return {
        ...base,
        title: "Unitree 真机预检未通过",
        message: "请检查真机网卡、机器人 IP、上游部署依赖和遥控器门控状态。",
        action: "preflight",
        actionLabel: "重新检查",
      };
    }
    if (/安全影子模式.*(?:未接入|尚未就绪)|execution_mode.{0,30}shadow/i.test(raw)) {
      return {
        ...base,
        tone: "warning",
        title: "安全影子模式尚未就绪",
        message: "只读真机状态链路尚未接入，当前不会访问真机或发送电机指令。",
      };
    }
    if (/lock|another .*running|already running|仍在运行|启动器.*运行/i.test(raw)) {
      return { ...base, title: "另一套仿真正在运行", message: "同一时间只能运行一套动捕动作链路，请先停止或清理当前流程。", action: "cleanup", actionLabel: "清理残留" };
    }
    if (/local state|resp2/i.test(raw)) {
      return { ...base, title: "Windows 本地状态服务不可用", message: "请关闭占用 127.0.0.1:6379 的其他程序后重试；Native 版本不需要另装 Redis。", action: "preflight", actionLabel: "重新检查" };
    }
    if (/redis/i.test(raw)) {
      return { ...base, title: "Windows 本地状态服务不可用", message: "Native 版本已内置状态服务，不需要安装 WSL 或 Redis；请关闭占用 127.0.0.1:6379 的其他程序后重试。", action: "preflight", actionLabel: "重新检查" };
    }
    if (/checkpoint|policy.*not found|model.*missing|模型.*缺失/i.test(raw)) {
      return { ...base, title: "控制模型文件缺失", message: "动作引擎无法加载所选模型，请检查项目安装是否完整。" };
    }
    if (/no valid|no .*frame|waiting.*udp|mocap|skeleton.*无|timeout.*frame/i.test(raw)) {
      return { ...base, title: "尚未收到动捕动作数据", message: "请确认动捕服务正在发送，并核对服务器 IP 和 Skeleton ID。" };
    }
    if (/segmentation|cmvrpn|libcmvrpn|chingmu sdk|青瞳 sdk/i.test(raw)) {
      return { ...base, title: "动捕接口启动失败", message: "动捕运行库需要在 Linux 工作目录中加载；日常使用请通过本控制台启动。" };
    }
    if (/cuda/i.test(raw)) {
      return { ...base, tone: "warning", title: "CUDA 当前不可用", message: "可改用 CPU 继续验证，但实时帧率可能降低。" };
    }
    if (/无法连接本地控制服务|响应超时/i.test(raw)) {
      return { ...base, title: "本地控制服务不可用", message: "请确认控制台启动窗口仍在运行，页面会继续自动重连。", action: "retry", actionLabel: "立即重试" };
    }
    return base;
  }

  function renderAlert() {
    let info = state.manualAlert;
    const status = state.status || {};
    const phase = normalizePhase(status.phase, status.running);
    const inputAge = finiteNumber(status.metrics?.input_age_ms ?? status.metrics?.packet_age_ms);

    if (status.error) {
      info = humanizeError(status.error, status.message || "");
    } else if (phase === "error" && !info) {
      info = humanizeError(status.message || "仿真流程异常结束", `exit_code=${status.exit_code ?? "unknown"}`);
    } else if (phase === "running" && inputAge !== null && inputAge > 750) {
      info = {
        tone: "danger",
        title: "动捕输入已中断",
        message: "数据延迟超过 750 ms，当前流程可能保持最后目标。真机请先使用实体急停，再停止流程并检查输入。",
        details: `input_age_ms=${inputAge.toFixed(1)}`,
        action: "stop",
        actionLabel: "停止流程（非急停）",
      };
    }

    if (!info) {
      elements.alertPanel.hidden = true;
      return;
    }

    elements.alertPanel.hidden = false;
    elements.alertPanel.dataset.tone = info.tone || "danger";
    elements.alertLabel.textContent = info.tone === "warning" ? "性能提示" : "需要处理";
    elements.alertTitle.textContent = info.title;
    elements.alertMessage.textContent = info.message;
    elements.alertDetails.textContent = info.details || "暂无更多技术详情";
    elements.alertDetailsWrapper.hidden = !info.details;
    elements.alertActionButton.dataset.action = info.action || "preflight";
    elements.alertActionButton.textContent = info.actionLabel || "重新检查";
  }

  async function handleAlertAction() {
    const action = elements.alertActionButton.dataset.action;
    if (action === "cleanup") return handleCleanup();
    if (action === "stop") return handleStop();
    if (action === "retry") return pollStatus(true);
    if (action === "copy-redis") {
      await copyText("sudo apt update && sudo apt install -y redis-server");
      showToast("Redis 安装命令已复制", "success");
      return;
    }
    return handleStandalonePreflight();
  }

  function applyStatus(status) {
    state.status = status || {};
    if (status.phase !== "error" && status.running) state.manualAlert = null;
    if (normalizePhase(status.phase, status.running) === "stopped") state.stopRequestedAt = null;

    const phase = normalizePhase(status.phase, status.running);
    const phaseInfo = PHASES[phase];
    if (phaseInfo.step >= 0 && phase !== "error") state.lastStep = phaseInfo.step;

    elements.stateCode.textContent = phaseInfo.code;
    elements.pipelineTitle.textContent = phaseInfo.title;
    elements.pipelineMessage.textContent = publicText(status.message || phaseInfo.message);
    elements.largeStatusIndicator.dataset.tone = phaseInfo.tone;
    elements.progressSummary.textContent = phaseInfo.summary;
    elements.lastUpdate.textContent = `最近更新 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;

    const duration = formatDuration(status.uptime_seconds);
    elements.runDuration.textContent = duration;
    elements.simulationDuration.textContent = duration;

    const backend = publicBackend(status.backend || state.config.backend || selectedBackend());
    const info = BACKENDS[backend];
    const executionMode = state.configDirty
      ? selectedExecutionMode()
      : state.config.execution_mode || selectedExecutionMode();
    elements.executionModeChip.textContent = EXECUTION_MODES[executionMode]?.label || EXECUTION_MODES.simulation_only.label;
    elements.activeModeChip.textContent = info.label;
    updateStepper(phase, phaseInfo.step);
    updateMetrics(status, phase, backend);
    renderSafetyStatus(status.safety);
    ingestLogs(status.logs);
    renderAlert();
    updateConfigLock();
    updateActionButtons();
    elements.operatorPrompt.hidden = phase !== "waiting_mocap";
    elements.liveStatus.textContent = publicText(`${phaseInfo.title}。${status.message || phaseInfo.message}`);
  }

  function formatSafetyAge(value) {
    const age = finiteNumber(value);
    if (age === null || age < 0) return "—";
    if (age < 1000) return `${Math.round(age)} ms`;
    return `${(age / 1000).toFixed(age < 10000 ? 1 : 0)} s`;
  }

  function renderSafetyStatus(rawSafety) {
    const safety = rawSafety && typeof rawSafety === "object" ? rawSafety : null;
    const available = Boolean(safety && Object.keys(safety).length && safety.available !== false);
    const configuredSafetyOff = state.config.safety_strategy_enabled === false;
    const realOutputConfigured = ["real_only", "parallel"].includes(state.config.execution_mode);

    if (!available) {
      const telemetryExpected = Boolean(state.status?.active)
        && normalizeBackend(state.status?.backend || state.config.backend) === "humanoid_gpt"
        && ["simulation_only", "parallel"].includes(state.config.execution_mode)
        && state.config.safety_strategy_enabled === true;
      const safetyOffTone = realOutputConfigured ? "warning" : "neutral";
      elements.safetyStatusCard.dataset.tone = configuredSafetyOff ? safetyOffTone : telemetryExpected ? "danger" : "neutral";
      setHealthBadge(
        elements.safetyHealth,
        configuredSafetyOff ? "二级安全策略 OFF" : telemetryExpected ? "遥测丢失" : "未接入",
        configuredSafetyOff ? safetyOffTone : telemetryExpected ? "danger" : "neutral"
      );
      elements.safetyGatewayState.textContent = configuredSafetyOff ? "UPSTREAM_DIRECT" : "NOT_CONNECTED";
      elements.safetyReason.textContent = String(
        configuredSafetyOff
          ? "实验性二级安全层未启用；当前使用上游原生链路，网页软件急停与复位不接入。"
          : safety?.reason || (telemetryExpected
          ? "运行中的实时生成式遥操作未提供可信 safety 状态。"
          : "后端尚未提供 safety 状态。")
      );
      elements.safetyReferenceHealth.textContent = "未接入";
      elements.safetyReferenceHealth.classList.remove("metric-warning", "metric-danger");
      elements.safetyReferenceAge.textContent = "—";
      elements.safetyReferenceAge.classList.remove("metric-warning", "metric-danger");
      elements.safetyTeleopWeight.textContent = "—";
      elements.safetyTeleopProgress.setAttribute("aria-valuenow", "0");
      elements.safetyTeleopProgress.setAttribute("aria-valuetext", "未接入");
      elements.safetyTeleopProgress.querySelector("span").style.width = "0%";
      elements.safetyEstopLatch.dataset.latched = configuredSafetyOff ? "false" : "unknown";
      elements.safetyEstopLatch.textContent = configuredSafetyOff ? "网页 E-STOP 不可用" : "E-STOP 状态未知";
      elements.safetyFaultLatch.dataset.latched = configuredSafetyOff ? "false" : "unknown";
      elements.safetyFaultLatch.textContent = configuredSafetyOff ? "二级 FAULT 锁未接入" : "FAULT 状态未知";
      elements.safetyStrategyState.textContent = configuredSafetyOff ? "OFF（默认）" : "—";
      elements.safetyStrategyState.classList.toggle("metric-warning", configuredSafetyOff && realOutputConfigured);
      elements.safetyStrategyState.classList.remove("metric-danger");
      elements.safetyHardGuards.textContent = configuredSafetyOff ? "未接入（上游）" : "—";
      elements.safetyHardGuards.classList.toggle("metric-warning", configuredSafetyOff && realOutputConfigured);
      elements.safetyHardGuards.classList.remove("metric-danger");
      elements.safetyLimited.textContent = configuredSafetyOff ? "不适用" : "—";
      elements.safetyLimited.classList.remove("metric-warning", "metric-danger");
      elements.dockSafetyTitle.textContent = configuredSafetyOff
        ? realOutputConfigured ? "实验性真机直通" : "原生仿真链路"
        : telemetryExpected ? "安全遥测丢失" : "安全状态未接入";
      elements.dockSafetyText.textContent = configuredSafetyOff
        ? realOutputConfigured
          ? "网页停止不是急停；必须使用实体遥控器门控与物理急停"
          : "实验性二级安全策略默认关闭；网页急停不接入"
        : telemetryExpected
        ? "按不健康状态处理；可锁存网页急停，但它不替代真机物理急停"
        : "等待后端安全遥测；网页急停不替代真机物理急停";
      if (configuredSafetyOff) state.softwareEstopLatched = false;
      return;
    }

    const gatewayState = String(
      safety.gateway_state
      ?? safety.state
      ?? safety.gateway?.state
      ?? "UNKNOWN"
    ).trim().toUpperCase();
    const reason = String(safety.reason ?? safety.message ?? safety.gateway?.reason ?? "未提供状态原因");
    const referenceHealth = String(
      safety.reference_health
      ?? safety.reference?.health
      ?? "UNKNOWN"
    ).trim().toUpperCase();
    let referenceAge = finiteNumber(safety.reference_age_ms ?? safety.reference?.age_ms);
    if (referenceAge === null) {
      const seconds = finiteNumber(safety.reference_age_s ?? safety.reference?.age_s);
      if (seconds !== null) referenceAge = seconds * 1000;
    }

    const rawWeight = finiteNumber(safety.teleop_weight ?? safety.gateway?.teleop_weight);
    const normalizedWeight = rawWeight === null
      ? null
      : Math.min(1, Math.max(0, rawWeight > 1 ? rawWeight / 100 : rawWeight));
    const weightPercent = normalizedWeight === null ? 0 : normalizedWeight * 100;
    const estopLatched = normalizeBoolean(
      safety.estop_latched ?? safety.latches?.estop,
      false
    );
    const faultLatched = normalizeBoolean(
      safety.fault_latched ?? safety.latches?.fault,
      false
    );
    const strategyValue = safety.safety_strategy_enabled
      ?? safety.strategy_enabled
      ?? safety.strategy?.enabled;
    const strategyEnabled = strategyValue === undefined || strategyValue === null
      ? state.config.safety_strategy_enabled !== false
      : normalizeBoolean(strategyValue, true);
    const legacyDirect = normalizeBoolean(
      safety.legacy_direct
      ?? safety.bypass_active
      ?? safety.bypassed
      ?? safety.strategy?.bypassed,
      false
    )
      || !strategyEnabled
      || ["LEGACY_DIRECT", "BYPASSED", "BYPASS_TRACKING"].includes(gatewayState);
    const displayGatewayState = legacyDirect ? "UPSTREAM_DIRECT" : gatewayState;
    let hardGuardsEnabled = optionalBoolean(
      safety.hard_guards_enabled ?? safety.hard_guards?.enabled
    );
    if (legacyDirect) hardGuardsEnabled = false;
    const limitedValue = safety.limited
      ?? safety.limit_active
      ?? safety.hard_limited
      ?? safety.limits?.active;
    const limitedCount = finiteNumber(safety.limited_count ?? safety.limits?.count);
    const limited = normalizeBoolean(limitedValue, false) || (limitedCount !== null && limitedCount > 0);

    let tone = "neutral";
    let badge = "已接入";
    let badgeTone = "active";
    if (legacyDirect) {
      tone = realOutputConfigured ? "warning" : "neutral";
      badge = "二级安全策略 OFF";
      badgeTone = realOutputConfigured ? "warning" : "neutral";
    } else if (estopLatched || faultLatched || ["E_STOP", "ESTOP", "FAULT"].includes(gatewayState)) {
      tone = "danger";
      badge = estopLatched ? "急停锁存" : "故障锁存";
      badgeTone = "danger";
    } else if (["INPUT_LOSS", "RECOVER_STAND"].includes(gatewayState)) {
      tone = "warning";
      badge = "正在恢复";
      badgeTone = "warning";
    } else if (gatewayState === "TELEOP") {
      tone = "ok";
      badge = "遥操作正常";
      badgeTone = "ok";
    } else if (gatewayState === "STANDBY") {
      tone = "warning";
      badge = "平衡待机";
      badgeTone = "warning";
    } else if (["ARMING", "BLEND_IN"].includes(gatewayState)) {
      badge = "正在过渡";
    } else if (gatewayState === "DISARMED") {
      badge = "未使能";
      badgeTone = "neutral";
    }
    if (hardGuardsEnabled === false && !legacyDirect) {
      tone = "danger";
      badge = "硬保护异常";
      badgeTone = "danger";
    } else if (hardGuardsEnabled === null && tone !== "danger") {
      tone = "warning";
      badge = "遥测不完整";
      badgeTone = "warning";
    }

    elements.safetyStatusCard.dataset.tone = tone;
    setHealthBadge(elements.safetyHealth, badge, badgeTone);
    elements.safetyGatewayState.textContent = displayGatewayState;
    elements.safetyReason.textContent = legacyDirect
      ? `实验性二级安全策略关闭，使用上游原生链路。${reason}`
      : state.config.execution_mode === "parallel"
        ? `仅保护并行中的仿真分支；真机不受此策略保护。${reason}`
        : reason;
    elements.safetyReferenceHealth.textContent = referenceHealth;
    elements.safetyReferenceHealth.classList.toggle(
      "metric-danger",
      ["LOST", "HARD_STALE", "INVALID", "ERROR"].includes(referenceHealth)
    );
    elements.safetyReferenceHealth.classList.toggle(
      "metric-warning",
      ["WAIT_FRESH", "SOFT_STALE", "STALE", "STANDBY", "DEGRADED"].includes(referenceHealth)
    );
    elements.safetyReferenceAge.textContent = formatSafetyAge(referenceAge);
    elements.safetyReferenceAge.classList.toggle(
      "metric-warning",
      referenceAge !== null && referenceAge >= 80 && referenceAge < 250
    );
    elements.safetyReferenceAge.classList.toggle(
      "metric-danger",
      referenceAge !== null && referenceAge >= 250
    );
    elements.safetyTeleopWeight.textContent = normalizedWeight === null
      ? "—"
      : `${weightPercent.toFixed(0)}%`;
    elements.safetyTeleopProgress.setAttribute("aria-valuenow", String(Math.round(weightPercent)));
    elements.safetyTeleopProgress.setAttribute(
      "aria-valuetext",
      normalizedWeight === null ? "未知" : `${weightPercent.toFixed(0)}%`
    );
    elements.safetyTeleopProgress.querySelector("span").style.width = `${weightPercent}%`;
    elements.safetyEstopLatch.dataset.latched = String(!legacyDirect && estopLatched);
    elements.safetyEstopLatch.textContent = legacyDirect
      ? "网页 E-STOP 不可用"
      : estopLatched ? "E-STOP 已锁存" : "E-STOP 未锁存";
    elements.safetyFaultLatch.dataset.latched = String(!legacyDirect && faultLatched);
    elements.safetyFaultLatch.textContent = legacyDirect
      ? "二级 FAULT 锁未接入"
      : faultLatched ? "FAULT 已锁存" : "FAULT 未锁存";
    elements.safetyStrategyState.textContent = legacyDirect ? "OFF（默认）" : "已启用（实验）";
    elements.safetyStrategyState.classList.toggle("metric-warning", legacyDirect && realOutputConfigured);
    elements.safetyStrategyState.classList.remove("metric-danger");
    elements.safetyHardGuards.textContent = legacyDirect
      ? "未接入（上游）"
      : hardGuardsEnabled === true
      ? "ON"
      : hardGuardsEnabled === false ? "异常关闭" : "未知";
    elements.safetyHardGuards.classList.toggle("metric-warning", !legacyDirect && hardGuardsEnabled === null);
    elements.safetyHardGuards.classList.toggle("metric-danger", !legacyDirect && hardGuardsEnabled === false);
    elements.safetyLimited.textContent = legacyDirect
      ? "不适用"
      : limitedCount !== null
      ? limitedCount > 0 ? `${Math.floor(limitedCount)} 次` : "否"
      : limited ? "是" : "否";
    elements.safetyLimited.classList.toggle("metric-warning", !legacyDirect && limited && hardGuardsEnabled === true);
    elements.safetyLimited.classList.toggle("metric-danger", !legacyDirect && hardGuardsEnabled === false);
    state.softwareEstopLatched = !legacyDirect && estopLatched;

    elements.dockSafetyTitle.textContent = legacyDirect
      ? realOutputConfigured ? "实验性真机直通" : "原生仿真链路"
      : state.config.execution_mode === "parallel"
        ? "仿真分支安全策略"
        : "实验性二级安全策略";
    elements.dockSafetyText.textContent = legacyDirect
      ? realOutputConfigured
        ? "网页停止不是急停；必须使用实体遥控器门控与物理急停"
        : "实验性二级安全策略默认关闭；网页急停不接入"
      : state.config.execution_mode === "parallel"
        ? "只保护仿真；真机仍为上游直通，网页停止不是急停"
        : "网页急停仅锁存仿真安全状态，不替代真机物理急停";
  }

  function updateStepper(phase, currentStep) {
    const steps = $$("li", elements.startupStepper);
    const effectiveStep = phase === "error" ? Math.max(0, state.lastStep) : currentStep;
    steps.forEach((step, index) => {
      step.classList.remove("is-complete", "is-current", "is-error");
      if (phase === "stopped") return;
      if (phase === "error" && index === effectiveStep) {
        step.classList.add("is-error");
      } else if (phase === "running") {
        step.classList.add(index < steps.length - 1 ? "is-complete" : "is-current");
      } else if (phase === "stopping") {
        step.classList.add("is-complete");
      } else if (index < effectiveStep) {
        step.classList.add("is-complete");
      } else if (index === effectiveStep) {
        step.classList.add("is-current");
      }
    });
  }

  function updateMetrics(status, phase, backend) {
    const metrics = status.metrics || {};
    const sourceFps = finiteNumber(metrics.source_fps ?? metrics.sender_fps);
    const bridgeFps = finiteNumber(metrics.bridge_fps);
    const backendFps = finiteNumber(metrics.backend_fps ?? metrics.policy_fps);
    const packetAge = finiteNumber(metrics.packet_age_ms);
    const inputAge = finiteNumber(metrics.input_age_ms ?? metrics.packet_age_ms);
    const rejected = Math.max(0, Math.floor(finiteNumber(metrics.rejected) || 0));
    const targetFps = finiteNumber(state.config.publish_fps) || 50;
    const active = ["waiting_mocap", "starting_backend", "running", "stopping"].includes(phase);
    const running = phase === "running";

    elements.sourceFps.textContent = formatNumber(sourceFps, 1);
    elements.metricSkeletonId.textContent = String(state.config.skeleton_id);
    elements.frameId.textContent = metrics.frame_id ?? "—";
    elements.captureFreshness.textContent = inputAge !== null
      ? inputAge < 1000 ? `${Math.round(inputAge)} ms 前` : `${(inputAge / 1000).toFixed(1)} s 前`
      : running ? "实时" : phase === "waiting_mocap" ? "等待中" : "—";

    if (inputAge !== null && inputAge > 750) setHealthBadge(elements.captureHealth, "输入已中断", "danger");
    else if (inputAge !== null && inputAge >= 100) setHealthBadge(elements.captureHealth, "数据延迟", "warning");
    else if (sourceFps !== null && sourceFps >= 45) setHealthBadge(elements.captureHealth, "数据正常", "ok");
    else if (sourceFps !== null && sourceFps > 0) setHealthBadge(elements.captureHealth, "帧率偏低", "warning");
    else if (phase === "waiting_mocap") setHealthBadge(elements.captureHealth, "等待数据", "active");
    else if (active) setHealthBadge(elements.captureHealth, "无新数据", "danger");
    else setHealthBadge(elements.captureHealth, "未连接", "neutral");

    elements.bridgeFps.textContent = formatNumber(bridgeFps, 1);
    elements.rootZ.textContent = finiteNumber(metrics.root_z) === null ? "—" : `${Number(metrics.root_z).toFixed(3)} m`;
    elements.rejectedFrames.textContent = String(rejected);
    elements.rejectedFrames.classList.toggle("metric-warning", rejected > 0);
    elements.anchorState.textContent = ["starting_backend", "running", "stopping"].includes(phase) ? "已建立" : "未建立";

    if (bridgeFps !== null && bridgeFps >= targetFps * 0.9 && bridgeFps <= targetFps * 1.2 && rejected === 0) {
      setHealthBadge(elements.gmrHealth, "重定向正常", "ok");
    } else if (bridgeFps !== null && bridgeFps >= targetFps * 0.7) {
      setHealthBadge(elements.gmrHealth, rejected > 0 ? "存在拒绝帧" : "频率波动", "warning");
    } else if (active && bridgeFps !== null) {
      setHealthBadge(elements.gmrHealth, "频率异常", "danger");
    } else if (active) {
      setHealthBadge(elements.gmrHealth, "正在启动", "active");
    } else {
      setHealthBadge(elements.gmrHealth, "未启动", "neutral");
    }

    const backendInfo = BACKENDS[backend];
    const preview = backend === "gmr_preview";
    elements.backendName.textContent = backendInfo.shortLabel;
    elements.backendDescription.textContent = backendInfo.description;
    elements.policyFps.textContent = preview
      ? "不适用"
      : backendFps === null ? `目标 ${targetFps} Hz` : `${backendFps.toFixed(1)} Hz`;
    elements.policyDevice.textContent = preview ? "不适用" : String(state.config.device || "auto").toUpperCase();
    elements.modelState.textContent = preview
      ? "不加载"
      : running ? "已加载" : phase === "starting_backend" ? "加载中" : "待加载";

    if (preview && running) setHealthBadge(elements.backendHealth, "仅预览", "ok");
    else if (running && (backendFps === null || backendFps >= targetFps * 0.7)) setHealthBadge(elements.backendHealth, "策略正常", "ok");
    else if (running) setHealthBadge(elements.backendHealth, "频率偏低", "warning");
    else if (phase === "starting_backend") setHealthBadge(elements.backendHealth, "正在加载", "active");
    else setHealthBadge(elements.backendHealth, preview ? "不适用" : "未加载", "neutral");

    elements.packetAge.textContent = formatNumber(packetAge, packetAge !== null && packetAge < 100 ? 1 : 0);
    elements.packetAge.classList.toggle("metric-warning", packetAge !== null && packetAge >= 100 && packetAge <= 500);
    elements.packetAge.classList.toggle("metric-danger", packetAge !== null && packetAge > 500);
    elements.viewerState.textContent = running
      ? preview ? "预览窗口已打开" : "已打开"
      : phase === "starting_backend" ? "正在打开" : "未打开";

    if (running && (packetAge === null || packetAge < 100)) setHealthBadge(elements.simulationHealth, preview ? "预览运行中" : "运行正常", "ok");
    else if (running && packetAge < 500) setHealthBadge(elements.simulationHealth, "延迟偏高", "warning");
    else if (running) setHealthBadge(elements.simulationHealth, "输入过期", "danger");
    else if (["starting", "starting_backend"].includes(phase)) setHealthBadge(elements.simulationHealth, "正在启动", "active");
    else if (phase === "stopping") setHealthBadge(elements.simulationHealth, "正在停止", "warning");
    else setHealthBadge(elements.simulationHealth, "未运行", "neutral");
  }

  function normalizeLogEntry(entry, index) {
    if (typeof entry === "string") {
      return {
        id: `text:${entry}`,
        timestamp: "",
        source: normalizeLogSource(entry.match(/^\[([^\]]+)\]/)?.[1]),
        level: /error|failed|traceback|exception/i.test(entry) ? "error" : /warn|waiting|rejected/i.test(entry) ? "warning" : "info",
        message: publicText(entry),
      };
    }
    const timestamp = entry.timestamp ?? entry.time ?? "";
    const component = entry.component ?? entry.source ?? "system";
    const rawMessage = String(entry.message ?? entry.text ?? "");
    return {
      id: String(entry.id ?? entry.seq ?? `${timestamp}:${component}:${rawMessage}:${index}`),
      timestamp,
      source: normalizeLogSource(component),
      level: normalizeLevel(entry.level),
      message: publicText(rawMessage),
    };
  }

  function ingestLogs(logs) {
    if (!logs) return;
    const entries = Array.isArray(logs) ? logs : String(logs).split(/\r?\n/).filter(Boolean);
    let changed = false;
    entries.forEach((entry, index) => {
      const normalized = normalizeLogEntry(entry, index);
      if (!normalized.message || state.seenLogIds.has(normalized.id) || state.suppressedLogIds.has(normalized.id)) return;
      state.seenLogIds.add(normalized.id);
      state.logs.push(normalized);
      changed = true;
    });

    if (state.logs.length > MAX_LOG_LINES) {
      const removed = state.logs.splice(0, state.logs.length - MAX_LOG_LINES);
      removed.forEach((entry) => state.seenLogIds.delete(entry.id));
    }
    if (changed && !state.logsPaused) renderLogs();
  }

  function filteredLogs() {
    const source = elements.logSourceFilter.value;
    const level = elements.logLevelFilter.value;
    return state.logs.filter((entry) => {
      return (source === "all" || entry.source === source) && (level === "all" || entry.level === level);
    });
  }

  function renderLogs() {
    const entries = filteredLogs();
    const fragment = document.createDocumentFragment();
    entries.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "log-line";
      row.dataset.level = entry.level;

      const time = document.createElement("span");
      time.className = "log-time";
      time.textContent = formatLogTime(entry.timestamp);
      const component = document.createElement("span");
      component.className = "log-component";
      component.textContent = sourceLabel(entry.source);
      const message = document.createElement("span");
      message.className = "log-message";
      message.textContent = entry.message;
      row.append(time, component, message);
      fragment.append(row);
    });

    elements.logLines.replaceChildren(fragment);
    elements.emptyLogs.hidden = entries.length > 0;
    elements.logCount.textContent = String(state.logs.length);

    const errors = state.logs.filter((entry) => entry.level === "error").length;
    const warnings = state.logs.filter((entry) => entry.level === "warning").length;
    if (errors) {
      elements.logSummary.textContent = `${errors} 个错误`;
      elements.logStatusDot.dataset.tone = "danger";
    } else if (warnings) {
      elements.logSummary.textContent = `${warnings} 个警告`;
      elements.logStatusDot.dataset.tone = "warning";
    } else {
      elements.logSummary.textContent = state.logs.length ? "运行记录正常" : "暂无消息";
      elements.logStatusDot.dataset.tone = "ok";
    }

    if (elements.autoScrollToggle.checked && elements.logsDetails.open) {
      requestAnimationFrame(() => {
        elements.logConsole.scrollTop = elements.logConsole.scrollHeight;
      });
    }
  }

  function clearLogsView() {
    state.logs.forEach((entry) => state.suppressedLogIds.add(entry.id));
    state.logs = [];
    state.seenLogIds.clear();
    renderLogs();
    showToast("已清空当前日志显示");
  }

  async function copyVisibleLogs() {
    const text = filteredLogs().map((entry) => `${formatLogTime(entry.timestamp)} [${sourceLabel(entry.source)}] ${entry.message}`).join("\n");
    if (!text) {
      showToast("当前没有可复制的日志");
      return;
    }
    await copyText(text);
    showToast("当前日志已复制", "success");
  }

  async function copyText(text) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  function toggleLogsPaused() {
    state.logsPaused = !state.logsPaused;
    elements.pauseLogsButton.textContent = state.logsPaused ? "继续显示" : "暂停显示";
    if (!state.logsPaused) renderLogs();
  }

  function showToast(message, tone = "neutral") {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.dataset.tone = tone;
    toast.textContent = message;
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 3800);
  }

  async function loadInitialConfig() {
    const payload = await fetchJson(API.config, {}, 6000);
    if (payload.options) applyOptions(payload.options);
    applyConfig(payload.config || payload, true);
  }

  async function pollStatus(manual = false) {
    if (state.consoleShutdown) return;
    if (state.polling) return;
    state.polling = true;
    if (state.pollTimer) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
    try {
      const payload = await fetchJson(API.status, {}, 3500);
      setServiceOnline(true);
      const status = extractStatus(payload) || payload;
      applyStatus(status);
    } catch (error) {
      setServiceOnline(false);
      if (manual) handleActionError(error);
    } finally {
      state.polling = false;
      state.pollTimer = window.setTimeout(() => pollStatus(false), POLL_INTERVAL_MS);
    }
  }

  function bindEvents() {
    elements.configForm.addEventListener("submit", handleSaveConfig);
    elements.configForm.addEventListener("input", (event) => {
      if (event.target.matches("input, select")) {
        setDirty(true);
        if (["mode", "execution_mode", "debug_mode", "safety_strategy_enabled"].includes(event.target.name) || event.target === elements.device) updateModePresentation();
      }
    });
    elements.configForm.addEventListener("change", (event) => {
      if (["mode", "execution_mode", "debug_mode", "safety_strategy_enabled"].includes(event.target.name) || event.target === elements.device) updateModePresentation();
    });

    elements.resetConfigButton.addEventListener("click", () => {
      applyConfig(DEFAULT_CONFIG, false);
      setDirty(true);
      showToast("已恢复默认值，保存后生效");
    });
    elements.preflightButton.addEventListener("click", handleStandalonePreflight);
    elements.startButton.addEventListener("click", handleStart);
    elements.stopButton.addEventListener("click", handleStop);
    elements.forceStopButton.addEventListener("click", handleForceStop);
    elements.emergencyStopButton.addEventListener("click", handleEmergencyStop);
    elements.safetyResetButton.addEventListener("click", handleSafetyReset);
    elements.cleanupButton.addEventListener("click", handleCleanup);
    elements.exitConsoleButton.addEventListener("click", handleExitConsole);
    elements.alertActionButton.addEventListener("click", handleAlertAction);
    elements.retryConnectionButton.addEventListener("click", () => pollStatus(true));

    elements.logSourceFilter.addEventListener("change", renderLogs);
    elements.logLevelFilter.addEventListener("change", renderLogs);
    elements.autoScrollToggle.addEventListener("change", renderLogs);
    elements.logsDetails.addEventListener("toggle", () => {
      if (elements.logsDetails.open) renderLogs();
    });
    elements.pauseLogsButton.addEventListener("click", toggleLogsPaused);
    elements.clearLogsButton.addEventListener("click", clearLogsView);
    elements.copyLogsButton.addEventListener("click", () => copyVisibleLogs().catch((error) => handleActionError(error)));

    window.addEventListener("beforeunload", (event) => {
      const phase = normalizePhase(state.status?.phase, state.status?.running);
      if (["starting", "waiting_mocap", "starting_backend", "running", "stopping"].includes(phase)) {
        event.preventDefault();
        event.returnValue = "";
      }
    });
  }

  async function initialize() {
    cacheElements();
    bindEvents();
    applyConfig(DEFAULT_CONFIG, true);
    applyStatus({ phase: "stopped", running: false, metrics: {}, logs: [] });
    void refreshProductLicense();

    const [configResult] = await Promise.allSettled([
      loadInitialConfig(),
      fetchJson(API.health, {}, 3500),
    ]);

    if (configResult.status === "rejected") {
      setServiceOnline(false);
    }
    await pollStatus(false);
  }

  document.addEventListener("DOMContentLoaded", initialize);
})();

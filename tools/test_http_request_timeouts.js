#!/usr/bin/env node
"use strict";

// Exercise the real frontend request helper without a browser, network,
// application bootstrap, or robot. All elapsed time is a simulated clock.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appPath = path.join(__dirname, "..", "src", "controller", "static", "app.js");
const bootstrap = '  document.addEventListener("DOMContentLoaded", initialize);';
const source = fs.readFileSync(appPath, "utf8");
assert.equal(source.split(bootstrap).length, 2, "frontend bootstrap anchor must be unique");
const testSource = source.replace(
  bootstrap,
  `  globalThis.__requestTest = { fetchJson };\n${bootstrap}`,
);

async function flushPromises() {
  // fetchJson awaits both fetch() and response.text(), then test observers run.
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
}

function harness() {
  let now = 0;
  let timerId = 0;
  const timers = new Map();
  const requests = [];
  const clock = {
    setTimeout(callback, delay) {
      const id = ++timerId;
      timers.set(id, { at: now + Number(delay), callback });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    async advance(milliseconds) {
      const end = now + milliseconds;
      for (;;) {
        const next = [...timers].sort((left, right) => left[1].at - right[1].at)[0];
        if (!next || next[1].at > end) break;
        const [id, timer] = next;
        now = timer.at;
        timers.delete(id);
        timer.callback();
        await flushPromises();
      }
      now = end;
      await flushPromises();
    },
  };

  const context = vm.createContext({
    AbortController,
    window: clock,
    document: {
      addEventListener(name, callback) {
        assert.equal(name, "DOMContentLoaded");
        assert.equal(typeof callback, "function");
        // Registering bootstrap must never start the application in this test.
      },
    },
    fetch(url, options) {
      return new Promise((resolve, reject) => {
        const request = {
          url,
          options,
          aborts: 0,
          respond(payload = { ok: true }, status = 200) {
            resolve({
              ok: status >= 200 && status < 300,
              status,
              text: async () => JSON.stringify(payload),
            });
          },
          fail(message = "Connection closed") {
            reject(new Error(message));
          },
        };
        const abort = () => {
          request.aborts += 1;
          const error = new Error("Request aborted");
          error.name = "AbortError";
          reject(error);
        };
        requests.push(request);
        if (options.signal.aborted) abort();
        else options.signal.addEventListener("abort", abort, { once: true });
      });
    },
  });
  vm.runInContext(testSource, context, { filename: appPath });
  assert.equal(requests.length, 0, "loading the frontend must not issue requests");

  function begin(url, options = {}, timeout) {
    const outcome = { state: "pending" };
    const promise = timeout === undefined
      ? context.__requestTest.fetchJson(url, options)
      : context.__requestTest.fetchJson(url, options, timeout);
    // Attach both observers before advancing time to avoid unhandled rejections.
    promise.then(
      (value) => Object.assign(outcome, { state: "fulfilled", value }),
      (error) => Object.assign(outcome, { state: "rejected", error }),
    );
    return outcome;
  }

  return { clock, timers, requests, begin };
}

for (const { label, url, elapsed, method } of [
  { label: "license response after 16 seconds", url: "/api/license?refresh=1", elapsed: 16000, method: "GET" },
  { label: "preflight response after 46 seconds", url: "/api/preflight", elapsed: 46000, method: "POST" },
  { label: "start response after 25 seconds", url: "/api/start", elapsed: 25000, method: "POST" },
]) {
  test(`${label} is not aborted prematurely`, async () => {
    const runtime = harness();
    const outcome = runtime.begin(url, { method });
    const request = runtime.requests[0];
    await runtime.clock.advance(elapsed);
    assert.equal(outcome.state, "pending");
    assert.equal(request.options.signal.aborted, false);
    assert.equal(request.aborts, 0);
    assert.equal(runtime.requests.length, 1);

    request.respond({ ok: true, completed: url });
    await flushPromises();
    assert.equal(outcome.state, "fulfilled");
    assert.equal(outcome.value.completed, url);
    assert.equal(runtime.timers.size, 0, "successful requests must clear their abort timer");
    await runtime.clock.advance(120000);
    assert.equal(request.aborts, 0, "a completed request must not be aborted later");
  });
}

for (const url of ["/api/preflight", "/api/start"]) {
  test(`${url} still times out at 120 seconds and is not retried`, async () => {
    const runtime = harness();
    const outcome = runtime.begin(url, { method: "POST" });
    await runtime.clock.advance(119999);
    assert.equal(outcome.state, "pending");
    await runtime.clock.advance(1);
    assert.equal(outcome.state, "rejected");
    assert.equal(outcome.error.name, "ApiError");
    assert.equal(outcome.error.message, "控制服务响应超时");
    assert.equal(outcome.error.details, `${url} 未在 120000 ms 内响应`);
    assert.equal(runtime.requests[0].aborts, 1);
    assert.equal(runtime.timers.size, 0);
    await runtime.clock.advance(240000);
    assert.equal(runtime.requests.length, 1, "timed-out commands must not be retried automatically");
  });
}

test("status polling retains its explicit 3.5 second timeout and timeout message", async () => {
  const runtime = harness();
  const outcome = runtime.begin("/api/status", {}, 3500);
  await runtime.clock.advance(3499);
  assert.equal(outcome.state, "pending");
  await runtime.clock.advance(1);
  assert.equal(outcome.state, "rejected");
  assert.equal(outcome.error.message, "控制服务响应超时");
  assert.equal(outcome.error.details, "/api/status 未在 3500 ms 内响应");
  assert.equal(runtime.requests[0].aborts, 1);
  assert.equal(runtime.timers.size, 0);
});

test("ordinary requests retain their default 6 second timeout", async () => {
  const runtime = harness();
  const outcome = runtime.begin("/api/config");
  await runtime.clock.advance(5999);
  assert.equal(outcome.state, "pending");
  await runtime.clock.advance(1);
  assert.equal(outcome.state, "rejected");
  assert.equal(outcome.error.details, "/api/config 未在 6000 ms 内响应");
});

for (const failure of ["network", "HTTP rejection"]) {
  test(`a start ${failure} is reported without an automatic retry`, async () => {
    const runtime = harness();
    const outcome = runtime.begin("/api/start", { method: "POST" });
    if (failure === "network") runtime.requests[0].fail();
    else runtime.requests[0].respond({ error: "Environment check failed" }, 412);
    await flushPromises();
    assert.equal(outcome.state, "rejected");
    assert.equal(outcome.error.name, "ApiError");
    assert.equal(
      outcome.error.message,
      failure === "network" ? "无法连接本地控制服务" : "Environment check failed",
    );
    assert.equal(outcome.error.status, failure === "network" ? 0 : 412);
    assert.equal(runtime.timers.size, 0);
    await runtime.clock.advance(240000);
    assert.equal(runtime.requests.length, 1, "failed start commands must not be retried automatically");
    assert.equal(runtime.requests[0].aborts, 0);
  });
}

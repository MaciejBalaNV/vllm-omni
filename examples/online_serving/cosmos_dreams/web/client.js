const video = document.querySelector("#video");
const connectButton = document.querySelector("#connect");
const resetButton = document.querySelector("#reset");
const disconnectButton = document.querySelector("#disconnect");
const statusNode = document.querySelector("#status");
const metricsNode = document.querySelector("#metrics");

const tracked = new Set(["1", "2", "3", "w", "s", "a", "d", "r", "f", "i", "k", "j", "l", "u", "o", "space"]);
const held = new Set();
let pc = null;
let channel = null;
let heartbeat = null;

function normalizedKey(event) {
  if (event.code === "Space") return "space";
  return String(event.key || "").toLowerCase();
}

function setStatus(message, error = false) {
  statusNode.textContent = message;
  statusNode.dataset.error = String(error);
}

function send(payload) {
  if (channel?.readyState === "open") channel.send(JSON.stringify(payload));
}

function sendKey(event, key) {
  send({type: "action", action: {event, key}});
}

function releaseAll() {
  for (const key of held) sendKey("keyup", key);
  held.clear();
}

async function connect() {
  connectButton.disabled = true;
  setStatus("Connecting…");
  pc = new RTCPeerConnection();
  pc.addTransceiver("video", {direction: "recvonly"});
  channel = pc.createDataChannel("controls", {ordered: true});
  channel.onopen = () => {
    setStatus("Connected — press a control key to start");
    resetButton.disabled = false;
    disconnectButton.disabled = false;
    heartbeat = setInterval(() => send({type: "heartbeat"}), 5000);
  };
  channel.onmessage = event => {
    const payload = JSON.parse(event.data);
    if (payload.type === "error") {
      setStatus(payload.message, true);
      return;
    }
    if (payload.type === "chunk_done") {
      setStatus(`Chunk ${payload.chunk_index} queued (${payload.frames} frames, depth ${payload.queue_depth})`);
      metricsNode.textContent = JSON.stringify(payload.timing, null, 2);
    }
    if (payload.type === "chunk_streaming") {
      const acknowledge = () => send({
        type: "presented",
        generation_id: payload.generation_id,
        chunk_index: payload.chunk_index,
      });
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(acknowledge);
      else requestAnimationFrame(acknowledge);
    }
    if (payload.type === "presentation_timing") {
      metricsNode.textContent = JSON.stringify(payload.timing, null, 2);
    }
  };
  pc.ontrack = event => { video.srcObject = event.streams[0]; };
  pc.onconnectionstatechange = () => {
    if (["failed", "closed", "disconnected"].includes(pc.connectionState)) disconnect(false);
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const response = await fetch("/offer", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type}),
  });
  if (!response.ok) {
    setStatus(await response.text(), true);
    await disconnect(false);
    return;
  }
  await pc.setRemoteDescription(await response.json());
}

async function disconnect(notify = true) {
  releaseAll();
  if (notify) send({type: "disconnect"});
  if (heartbeat) clearInterval(heartbeat);
  heartbeat = null;
  channel?.close();
  channel = null;
  if (pc) await pc.close();
  pc = null;
  video.srcObject = null;
  connectButton.disabled = false;
  resetButton.disabled = true;
  disconnectButton.disabled = true;
  setStatus("Disconnected");
}

window.addEventListener("keydown", event => {
  const key = normalizedKey(event);
  if (!tracked.has(key) || event.repeat || held.has(key)) return;
  event.preventDefault();
  held.add(key);
  sendKey("keydown", key);
});
window.addEventListener("keyup", event => {
  const key = normalizedKey(event);
  if (!tracked.has(key) || !held.has(key)) return;
  event.preventDefault();
  held.delete(key);
  sendKey("keyup", key);
});
window.addEventListener("blur", releaseAll);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAll(); });
connectButton.addEventListener("click", () => connect().catch(error => setStatus(String(error), true)));
resetButton.addEventListener("click", () => { releaseAll(); send({type: "reset"}); });
disconnectButton.addEventListener("click", () => disconnect(true));

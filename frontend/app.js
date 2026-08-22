// app.js — shared API client + auth helpers used by every page.

const API_BASE = window.API_BASE || "http://localhost:8000";

function getToken() {
  return localStorage.getItem("campus_token");
}
function setToken(t) {
  localStorage.setItem("campus_token", t);
}
function clearToken() {
  localStorage.removeItem("campus_token");
}
function isLoggedIn() {
  return !!getToken();
}

function requireAuth() {
  if (!isLoggedIn()) {
    window.location.href = "login.html";
  }
}

async function api(path, { method = "GET", body, isForm = false, auth = true } = {}) {
  const headers = {};
  if (auth && getToken()) headers["Authorization"] = `Bearer ${getToken()}`;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    // no JSON body (e.g. 204)
  }

  if (!res.ok) {
    const detail = data && data.detail ? data.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function login(username, password) {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);

  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed");
  setToken(data.access_token);
  return data;
}

async function getCurrentUser() {
  return api("/auth/me");
}

function showMsg(el, text, type = "info") {
  el.textContent = text;
  el.className = `msg ${type}`;
  el.style.display = "block";
}

function renderNav(activePage) {
  const el = document.getElementById("nav");
  if (!el) return;
  const pages = [
    ["dashboard.html", "Dashboard"],
    ["attendance.html", "Attendance"],
    ["chatbot.html", "Assistant"],
    ["students.html", "Students"],
  ];
  el.innerHTML = pages
    .map(
      ([href, label]) =>
        `<a href="${href}" class="${activePage === href ? "active" : ""}">${label}</a>`
    )
    .join("") + `<a href="#" id="logoutLink">Logout</a>`;

  document.getElementById("logoutLink").addEventListener("click", (e) => {
    e.preventDefault();
    clearToken();
    window.location.href = "login.html";
  });
}

// Captures a single frame from a <video> element as a JPEG Blob.
function captureFrameAsBlob(videoEl) {
  return new Promise((resolve) => {
    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    canvas.getContext("2d").drawImage(videoEl, 0, 0);
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
  });
}

async function startWebcam(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoEl.srcObject = stream;
  await videoEl.play();
  return stream;
}

function stopWebcam(stream) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
}

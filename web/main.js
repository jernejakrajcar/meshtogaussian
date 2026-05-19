import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { createRawGaussianRenderer } from "./raw_gaussian_renderer.js";
import { depthSortedOrder, reorderByOrder } from "./splat_sort.js";

const modelSelect = document.querySelector("#modelSelect");
const uploadInput = document.querySelector("#uploadInput");
const refreshModelsButton = document.querySelector("#refreshModelsButton");
const prepareButton = document.querySelector("#prepareButton");
const convertMesh2SplatButton = document.querySelector("#convertMesh2SplatButton");
const mesh2splatDensity = document.querySelector("#mesh2splatDensity");
const representationSelect = document.querySelector("#representationSelect");
const trainedSelect = document.querySelector("#trainedSelect");
const modeSelect = document.querySelector("#modeSelect");
const transitionStyleSelect = document.querySelector("#transitionStyleSelect");
const backgroundSelect = document.querySelector("#backgroundSelect");
const transitionSlider = document.querySelector("#transitionSlider");
const transitionValue = document.querySelector("#transitionValue");
const lockTransitionViewButton = document.querySelector("#lockTransitionViewButton");
const lockCameraButton = document.querySelector("#lockCameraButton");
const lodSelect = document.querySelector("#lodSelect");
const pointSize = document.querySelector("#pointSize");
const opacity = document.querySelector("#opacity");
const gaussianYOffset = document.querySelector("#gaussianYOffset");
const gaussianScale = document.querySelector("#gaussianScale");
const statusBox = document.querySelector("#status");
const viewer = document.querySelector("#viewer");
const loadingOverlay = document.querySelector("#loadingOverlay");
const loadingMessage = document.querySelector("#loadingMessage");
const representationHint = document.querySelector("#representationHint");
const trainedHint = document.querySelector("#trainedHint");
const convertHint = document.querySelector("#convertHint");
const lodHint = document.querySelector("#lodHint");
const transitionStyleHint = document.querySelector("#transitionStyleHint");
const transitionHint = document.querySelector("#transitionHint");
const lockTransitionHint = document.querySelector("#lockTransitionHint");
const lockCameraHint = document.querySelector("#lockCameraHint");
const uiControls = [...document.querySelectorAll(".sidebar button, .sidebar input, .sidebar select")];
const APP_VERSION = "splat-render-3-raw-webgl";
const DEFAULT_TRAINED_LOD_COUNTS = [
  10, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000, 100000,
];

const state = {
  preparedId: null,
  prepared: null,
  meshObject: null,
  selectedGaussianObject: null,
  transitionObjects: new Map(),
  sortedTransitionObjects: new Map(),
  sortedTransitionStats: new Map(),
  lodCache: new Map(),
  transitionRequestId: 0,
  busy: false,
  busyStartedAt: 0,
  busyTimer: null,
  meshStatus: "",
  transitionCameraUpdateQueued: false,
  transitionViewLock: null,
  autoSortView: null,
  cameraLock: {
    enabled: false,
    automatic: false,
    userUnlockedAuto: false,
  },
};
const MESH_OPACITY = 1.0;
const SORT_LOADING_THRESHOLD = 300000;
const AUTO_SORT_THRESHOLD = 120000;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2.2, 1.3, 2.6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
viewer.appendChild(renderer.domElement);
const gaussianCanvas = document.createElement("canvas");
gaussianCanvas.id = "gaussianLayer";
viewer.appendChild(gaussianCanvas);
const rawGaussianRenderer = createRawGaussianRenderer(gaussianCanvas);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const gltfLoader = new GLTFLoader();

const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(2.0, 4.0, 3.0);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.25);
fillLight.position.set(-3.0, 2.0, 2.0);
scene.add(fillLight);
scene.add(new THREE.HemisphereLight(0xffffff, 0x9fb0c0, 0.55));
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const grid = new THREE.GridHelper(2.4, 12, 0x46515c, 0x252c33);
scene.add(grid);

function setStatus(message) {
  const context = buildStatusContext();
  const detail = state.meshStatus ? `\n\n${state.meshStatus}` : "";
  statusBox.textContent = `${context}${message ? `\n\n${message}` : ""}${detail}\n\nViewer script: ${APP_VERSION}`;
  statusBox.scrollTop = statusBox.scrollHeight;
}

function setMeshStatus(message) {
  state.meshStatus = message;
}

function buildStatusContext() {
  const selectedModel = modelSelect.options[modelSelect.selectedIndex]?.textContent || "none";
  const prepared = state.prepared;
  const representation = prepared ? representationLabel(prepared.representation) : representationLabel(representationSelect.value);
  const gaussianSource = prepared?.gaussian_source || "not loaded";
  const lodCounts = prepared?.lods?.map((lod) => `${lod.name}:${lod.count.toLocaleString()}`).join(", ") || "not loaded";
  const lockState = state.transitionViewLock
    ? `locked (${state.sortedTransitionObjects.size} sorted object(s))`
    : "unlocked";
  const currentLod = lodSelect.value || "none";
  return [
    "Setup",
    `model: ${selectedModel}`,
    `prepared: ${prepared ? representation : "not loaded"}`,
    `gaussian source: ${gaussianSource}`,
    `lods: ${lodCounts}`,
    `view mode: ${viewModeLabel(modeSelect.value)}`,
    `transition lock: ${lockState}`,
    `current LOD: ${currentLod}`,
  ].join("\n");
}

function representationLabel(value) {
  if (value === "mesh2splat_lods") return "Mesh2Splat LOD files";
  if (value === "trained") return "Single trained PLY";
  if (value === "initialized") return "Mesh-sampled preview";
  return value || "unknown";
}

function viewModeLabel(value) {
  if (value === "transition") return "Transition";
  if (value === "both") return "Mesh + LOD";
  if (value === "mesh") return "Mesh";
  if (value === "gaussian") return "Selected Gaussian LOD";
  return value || "unknown";
}

function setHint(element, text, disabled = false) {
  if (!element) return;
  element.textContent = text;
  element.classList.toggle("is-disabled", disabled);
}

function setControlAvailability(control, enabled, reason = "") {
  if (!control) return;
  control.disabled = !enabled;
  control.title = enabled ? "" : reason;
  const wrapper = control.closest?.(".control");
  if (!wrapper) return;
  if (enabled || !reason) wrapper.removeAttribute("data-disabled-reason");
  else wrapper.dataset.disabledReason = reason;
}

function setBusy(active, message = "Preparing viewer data...") {
  state.busy = active;
  if (state.busyTimer) {
    clearInterval(state.busyTimer);
    state.busyTimer = null;
  }

  for (const control of uiControls) control.disabled = active;
  loadingOverlay.hidden = !active;

  if (active) {
    state.busyStartedAt = performance.now();
    updateBusyMessage(message);
    state.busyTimer = setInterval(() => updateBusyMessage(message), 500);
    return;
  }

  updateControlAvailability();
}

function updateBusyMessage(message) {
  const elapsed = Math.max(0, (performance.now() - state.busyStartedAt) / 1000);
  const text = `${message}\nElapsed: ${elapsed.toFixed(1)}s`;
  loadingMessage.textContent = text;
  setStatus(text);
}

function updateControlAvailability() {
  if (state.busy) return;
  for (const control of uiControls) setControlAvailability(control, true);
  const hasPrepared = Boolean(state.prepared);
  const inTransition = modeSelect.value === "transition";
  const inLodView = modeSelect.value === "gaussian" || modeSelect.value === "both";
  const usesTrainedPly = representationSelect.value === "trained";
  const isDemoModel = modelSelect.value?.startsWith("demo:");

  setControlAvailability(trainedSelect, usesTrainedPly, "Only used for Single trained PLY.");
  setControlAvailability(transitionStyleSelect, inTransition, "Only used in Transition mode.");
  setControlAvailability(transitionSlider, inTransition, "Only used in Transition mode.");
  setControlAvailability(lodSelect, hasPrepared && inLodView, inTransition
    ? "Disabled in Transition mode because the transition blends LODs automatically."
    : "Load a setup and choose Mesh + LOD or Selected Gaussian LOD.");
  setControlAvailability(lockTransitionViewButton, hasPrepared && inTransition, "Only for Transition mode; depth-sorts splats for current camera.");
  setControlAvailability(lockCameraButton, hasPrepared && inLodView, "For Gaussian/Both view with trained splats.");
  setControlAvailability(convertMesh2SplatButton, !isDemoModel, "Mesh2Splat conversion requires a real mesh file.");
  setControlAvailability(mesh2splatDensity, !isDemoModel, "Mesh2Splat conversion requires a real mesh file.");

  updateInlineHints();
  updateTransitionLockUi();
  updateCameraLockUi();
}

function updateInlineHints() {
  const sourceHints = {
    mesh2splat_lods: "Uses matching files from data/mesh2splats.",
    trained: "Builds viewer LODs from one trained .ply.",
    initialized: "Fallback preview, not trained splats.",
  };
  const usesTrainedPly = representationSelect.value === "trained";
  const inTransition = modeSelect.value === "transition";
  const inLodView = modeSelect.value === "gaussian" || modeSelect.value === "both";
  const hasPrepared = Boolean(state.prepared);
  const isDemoModel = modelSelect.value?.startsWith("demo:");

  setHint(representationHint, sourceHints[representationSelect.value] ?? "");
  setHint(trainedHint, usesTrainedPly ? "Builds viewer LODs from one trained .ply." : "Only used for Single trained PLY.", !usesTrainedPly);
  setHint(convertHint, isDemoModel ? "Mesh2Splat conversion requires a real mesh file." : "Creates a new single trained PLY from the selected mesh.", isDemoModel);
  setHint(lodHint, inLodView && hasPrepared ? "Select the loaded Gaussian LOD for Mesh + LOD or Selected Gaussian LOD." : "Disabled in Transition mode because the transition blends LODs automatically.", !(inLodView && hasPrepared));
  setHint(transitionStyleHint, inTransition ? "Controls how mesh and splats are blended during Transition mode." : "Only used in Transition mode.", !inTransition);
  setHint(transitionHint, inTransition ? "Move camera distance through the configured transition range." : "Only used in Transition mode.", !inTransition);
  setHint(lockTransitionHint, inTransition ? "Only for Transition mode; depth-sorts splats for current camera." : "Only for Transition mode; depth-sorts splats for current camera.", !inTransition);
  setHint(lockCameraHint, inLodView ? "For Gaussian/Both view with trained splats." : "For Gaussian/Both view with trained splats.", !inLodView);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function smoothstep(edge0, edge1, x) {
  if (edge0 === edge1) return x >= edge1 ? 1 : 0;
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function lodSortKey(name) {
  const value = Number(name);
  return Number.isFinite(value) ? value : -1;
}

function transitionWeights(distance, transition) {
  const meshWeight = 1 - smoothstep(transition.mesh_fade_start ?? 3.6, transition.mesh_fade_end ?? 2.2, distance);
  if (transition.lod_mode === "progressive") {
    return progressiveTransitionWeights(meshWeight, transition);
  }
  const raw = {};
  for (const [name, range] of Object.entries(transition.lod_ranges ?? {})) {
    const far = Number(range[0]);
    const near = Number(range[1]);
    const enters = smoothstep(far, near, distance);
    const exits = near > 0 ? smoothstep(near, near * 0.72, distance) : 0;
    raw[name] = Math.max(0, enters * (1 - exits));
  }

  const gaussianBudget = Math.max(0, 1 - meshWeight);
  const rawTotal = Object.values(raw).reduce((sum, value) => sum + value, 0);
  const lods = {};
  if (rawTotal <= 1e-8) {
    for (const name of Object.keys(raw)) lods[name] = 0;
    if (gaussianBudget > 0 && Object.keys(raw).length > 0) {
      const closest = Object.keys(raw).sort((a, b) => lodSortKey(a) - lodSortKey(b)).at(-1);
      lods[closest] = gaussianBudget;
    }
  } else {
    for (const [name, value] of Object.entries(raw)) lods[name] = (gaussianBudget * value) / rawTotal;
  }

  const total = meshWeight + Object.values(lods).reduce((sum, value) => sum + value, 0);
  if (total <= 1e-8) return { mesh: 1, gaussian_lods: lods };
  for (const name of Object.keys(lods)) lods[name] /= total;
  return { mesh: meshWeight / total, gaussian_lods: lods };
}

function progressiveTransitionWeights(meshWeight, transition) {
  const names = Object.keys(transition.lod_ranges ?? {}).sort((a, b) => lodSortKey(a) - lodSortKey(b));
  const lods = {};
  for (const name of names) lods[name] = 0;
  if (names.length === 0) return { mesh: 1, gaussian_lods: lods };

  const gaussianBudget = Math.max(0, Math.min(1, 1 - meshWeight));
  if (gaussianBudget <= 1e-5) return { mesh: 1, gaussian_lods: lods };

  const densityT = Math.sqrt(gaussianBudget);
  const lodPosition = densityT * (names.length - 1);
  const lowerIndex = Math.floor(lodPosition);
  const upperIndex = Math.min(names.length - 1, lowerIndex + 1);
  const mix = lodPosition - lowerIndex;
  lods[names[lowerIndex]] += gaussianBudget * (1 - mix);
  lods[names[upperIndex]] += gaussianBudget * mix;
  return { mesh: Math.max(0, Math.min(1, meshWeight)), gaussian_lods: lods };
}

function additiveTransitionWeights(distance, transition) {
  const lods = {};
  for (const [name, range] of Object.entries(transition.lod_ranges ?? {})) {
    const far = Number(range[0]);
    const near = Number(range[1]);
    lods[name] = smoothstep(far, near, distance);
  }
  return { mesh: 1, gaussian_lods: lods };
}

function denseCutoverTransitionWeights(t, transition) {
  const names = Object.keys(transition.lod_ranges ?? {}).sort((a, b) => lodSortKey(a) - lodSortKey(b));
  const lods = {};
  for (const name of names) lods[name] = 0;
  if (!names.length) return { mesh: 1, gaussian_lods: lods };
  const denseName = names.at(-1);
  const gaussian = Math.max(0, Math.min(1, (t - 0.45) / 0.55));
  lods[denseName] = gaussian;
  return { mesh: 1 - gaussian, gaussian_lods: lods };
}

function detailBuildTransitionWeights(t, transition) {
  const names = Object.keys(transition.lod_ranges ?? {}).sort((a, b) => lodSortKey(a) - lodSortKey(b));
  const lods = {};
  for (const name of names) lods[name] = 0;
  if (!names.length) return { mesh: 1, gaussian_lods: lods };

  const detailT = smoothstep(0.08, 0.88, t);
  const lodPosition = detailT * (names.length - 1);
  const lowerIndex = Math.floor(lodPosition);
  const upperIndex = Math.min(names.length - 1, lowerIndex + 1);
  const mix = lodPosition - lowerIndex;
  const strength = smoothstep(0.05, 0.32, t);
  lods[names[lowerIndex]] += strength * (1 - mix);
  lods[names[upperIndex]] += strength * mix;

  const mesh = 1 - smoothstep(0.88, 0.98, t);
  return { mesh, gaussian_lods: lods };
}

function applyBackgroundTheme() {
  const light = backgroundSelect.value === "light";
  scene.background = new THREE.Color(light ? 0xf4f1ea : 0x111418);
  const colors = light ? [0x79818a, 0xc9c3b8] : [0x46515c, 0x252c33];
  const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
  for (const material of materials) {
    material.color.setHex(colors[0]);
    material.opacity = light ? 0.45 : 1.0;
    material.transparent = light;
    material.needsUpdate = true;
  }
}

function resize() {
  const rect = viewer.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  renderer.setSize(rect.width, rect.height);
  rawGaussianRenderer.setSize(rect.width, rect.height, pixelRatio);
  camera.aspect = rect.width / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
  updateGaussianViewportUniforms();
}

function disposeObject(object) {
  if (!object) return;
  if (object.userData?.kind === "gaussian" && object.dispose) {
    rawGaussianRenderer.disposeLayer(object);
    return;
  }
  scene.remove(object);
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    disposeMaterial(child.material);
  });
}

function emptyMeshPlaceholder(message) {
  const group = new THREE.Group();
  group.userData.kind = "mesh";
  group.userData.textured = false;
  group.userData.loadFailed = true;
  group.userData.message = message;
  return group;
}

function disposeMaterial(material) {
  if (!material) return;
  const materials = Array.isArray(material) ? material : [material];
  for (const item of materials) {
    for (const value of Object.values(item)) {
      if (value?.isTexture) value.dispose();
    }
    item.dispose?.();
  }
}

function allGaussianObjects() {
  const objects = [
    state.selectedGaussianObject,
    ...state.transitionObjects.values(),
    ...state.sortedTransitionObjects.values(),
  ].filter(Boolean);
  return objects;
}

function clearTransitionObjects() {
  for (const object of state.transitionObjects.values()) disposeObject(object);
  state.transitionObjects.clear();
  clearSortedTransitionObjects();
}

function clearSortedTransitionObjects() {
  for (const object of state.sortedTransitionObjects.values()) disposeObject(object);
  state.sortedTransitionObjects.clear();
  state.sortedTransitionStats.clear();
}

function buildMesh(mesh) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(mesh.colors.flat(), 3));
  geometry.setIndex(mesh.faces.flat());
  geometry.computeVertexNormals();
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: MESH_OPACITY < 0.999,
    opacity: MESH_OPACITY,
  });
  return new THREE.Mesh(geometry, material);
}

async function buildSceneMesh(mesh) {
  const extension = mesh.source_extension?.toLowerCase();
  if (mesh.source_url && (extension === ".glb" || extension === ".gltf")) {
    try {
      setStatus(`Loading textured ${extension.toUpperCase()} mesh...`);
      const gltf = await gltfLoader.loadAsync(mesh.source_url);
      const root = gltf.scene;
      const radius = Number(mesh.radius) || 1;
      const center = mesh.center ?? [0, 0, 0];
      let texturedMaterialCount = 0;
      root.position.set(-center[0] / radius, -center[1] / radius, -center[2] / radius);
      root.scale.setScalar(1 / radius);
      root.traverse((child) => {
        if (!child.isMesh) return;
        child.castShadow = false;
        child.receiveShadow = false;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        const configuredMaterials = materials.map((material) => {
          const configured = configureLoadedMaterial(material);
          if (material.map) texturedMaterialCount += 1;
          return configured;
        });
        child.material = Array.isArray(child.material) ? configuredMaterials : configuredMaterials[0];
      });
      root.userData.kind = "mesh";
      root.userData.textured = true;
      applyMeshOpacity(root, MESH_OPACITY);
      setMeshStatus(`Mesh display: textured ${extension.toUpperCase()} loaded from ${mesh.source_url} (${texturedMaterialCount} base-color texture material(s)).`);
      setStatus("Textured mesh loaded.");
      return root;
    } catch (error) {
      const message = `Textured mesh load failed. Fallback color mesh is disabled for GLB/GLTF so this issue is visible:\n${error.message}`;
      setMeshStatus(`Mesh display: FAILED textured ${extension.toUpperCase()} load. No fallback mesh is being drawn.`);
      setStatus(message);
      return emptyMeshPlaceholder(message);
    }
  }
  const fallback = buildMesh(mesh);
  fallback.userData.kind = "mesh";
  fallback.userData.textured = false;
  setMeshStatus(`Mesh display: fallback vertex-color mesh (${mesh.source_extension ?? "generated"}).`);
  return fallback;
}

function configureLoadedMaterial(material) {
  const basicMaterial = new THREE.MeshBasicMaterial({
    alphaMap: material.alphaMap ?? null,
    alphaTest: material.alphaTest ?? 0,
    color: material.color?.clone?.() ?? new THREE.Color(0xffffff),
    map: material.map ?? null,
    opacity: MESH_OPACITY,
    side: THREE.DoubleSide,
    transparent: MESH_OPACITY < 0.999 || Boolean(material.transparent && material.opacity < 0.999) || Boolean(material.alphaMap),
    vertexColors: Boolean(material.vertexColors),
  });
  basicMaterial.userData.hasAlpha = Boolean(material.alphaMap || material.transparent);
  if (material.map) {
    basicMaterial.map.colorSpace = THREE.SRGBColorSpace;
    basicMaterial.map.needsUpdate = true;
    basicMaterial.color.set(0xffffff);
  }
  basicMaterial.depthWrite = true;
  basicMaterial.needsUpdate = true;
  return basicMaterial;
}

function applyMeshOpacity(object, value) {
  if (!object) return;
  object.traverse?.((child) => {
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    for (const material of materials) {
      material.opacity = value;
      material.transparent = value < 0.999 || Boolean(material.userData?.hasAlpha);
      material.needsUpdate = true;
    }
  });
  if (object.material) {
    object.material.opacity = value;
    object.material.transparent = value < 0.999;
    object.material.needsUpdate = true;
  }
}

function setTransitionControlsEnabled(enabled) {
  if (state.busy) return;
  const active = enabled && modeSelect.value === "transition";
  setControlAvailability(transitionSlider, active, "Only used in Transition mode.");
  setControlAvailability(transitionStyleSelect, active, "Only used in Transition mode.");
  setControlAvailability(lockTransitionViewButton, active && Boolean(state.prepared), "Only for Transition mode; depth-sorts splats for current camera.");
}

function shouldAutoLockCamera() {
  return state.prepared?.representation === "trained" && (modeSelect.value === "gaussian" || modeSelect.value === "both");
}

function applyControlsLockState() {
  controls.enabled = !state.transitionViewLock && !state.cameraLock.enabled;
}

function syncCameraLockForMode() {
  const shouldLock = shouldAutoLockCamera();
  if (shouldLock && !state.cameraLock.userUnlockedAuto) {
    state.cameraLock.enabled = true;
    state.cameraLock.automatic = true;
  } else if (!shouldLock && state.cameraLock.automatic) {
    state.cameraLock.enabled = false;
    state.cameraLock.automatic = false;
    state.cameraLock.userUnlockedAuto = false;
  }
  applyControlsLockState();
  updateCameraLockUi();
}

function toggleCameraLock() {
  if (!state.prepared) return;
  if (state.cameraLock.enabled) {
    state.cameraLock.enabled = false;
    state.cameraLock.automatic = false;
    state.cameraLock.userUnlockedAuto = shouldAutoLockCamera();
  } else {
    state.cameraLock.enabled = true;
    state.cameraLock.automatic = false;
    state.cameraLock.userUnlockedAuto = false;
  }
  applyControlsLockState();
  updateCameraLockUi();
}

function updateCameraLockUi() {
  if (!lockCameraButton) return;
  const inLodView = modeSelect.value === "gaussian" || modeSelect.value === "both";
  setControlAvailability(lockCameraButton, !state.busy && Boolean(state.prepared) && inLodView, "For Gaussian/Both view with trained splats.");
  const suffix = state.cameraLock.automatic ? " (auto)" : "";
  lockCameraButton.textContent = state.cameraLock.enabled ? `Unlock camera${suffix}` : "Lock camera";
}

function pointSizeMultiplier() {
  return Number(pointSize.value);
}

function gaussianPointScale() {
  return pointSizeMultiplier() * 8.0;
}

function gaussianGlobalScale() {
  const scale = Number(gaussianScale?.value ?? 1);
  return Number.isFinite(scale) ? scale : 1;
}

function gaussianGlobalYOffset() {
  const offset = Number(gaussianYOffset?.value ?? 0);
  return Number.isFinite(offset) ? offset : 0;
}

function applyGaussianTransform(object) {
  if (!object) return;
  object.userData.yOffset = gaussianGlobalYOffset();
  object.userData.scale = gaussianGlobalScale();
}

function rendererViewport() {
  const size = new THREE.Vector2();
  renderer.getDrawingBufferSize(size);
  return size;
}

function buildGaussianPoints(lod) {
  const points = rawGaussianRenderer.uploadScene(lod);
  points.userData.kind = "gaussian";
  points.opacityMultiplier = Number(opacity.value);
  return points;
}

function updateGaussianViewportUniforms() {
  rendererViewport();
}

function applyGaussianMaterial(object, opacityValue = Number(opacity.value)) {
  if (!object) return;
  applyGaussianTransform(object);
  object.opacityMultiplier = Math.max(0, Math.min(1, Number(opacityValue) || 0));
}

function drawRawGaussianLayer() {
  const viewport = rendererViewport();
  const visibleObjects = allGaussianObjects().filter((object) => object.visible && object.opacityMultiplier > 0);
  gaussianCanvas.hidden = visibleObjects.length === 0;
  rawGaussianRenderer.draw({
    camera,
    viewport: [viewport.x, viewport.y],
    opacity: 1,
    pointScale: gaussianPointScale(),
    gaussianScale: gaussianGlobalScale(),
    yOffset: gaussianGlobalYOffset(),
    clear: true,
  });
}

function hideGaussianObject(object) {
  if (!object) return;
  object.visible = false;
  applyGaussianMaterial(object, 0);
}

function hideAllGaussians() {
  hideGaussianObject(state.selectedGaussianObject);
  for (const object of state.transitionObjects.values()) hideGaussianObject(object);
  for (const object of allGaussianObjects()) hideGaussianObject(object);
}

async function getLod(count) {
  const key = `${state.preparedId}:${count}`;
  let lod = state.lodCache.get(key);
  if (!lod) {
    lod = await api(`/api/model/${state.preparedId}/lod/${count}`);
    state.lodCache.set(key, lod);
  }
  return lod;
}

function currentTransitionViewKey() {
  return state.transitionViewLock?.key ?? state.autoSortView?.key ?? "free";
}

function makeTransitionViewKey(position, target) {
  const values = [...position.toArray(), ...target.toArray()];
  return values.map((value) => value.toFixed(3)).join(":");
}

function currentSortViewMatrix() {
  return state.transitionViewLock?.viewMatrix ?? state.autoSortView?.viewMatrix ?? null;
}

function sortedLodForCurrentView(lod) {
  const viewMatrix = currentSortViewMatrix();
  if (!viewMatrix) return { lod, sortMs: 0 };
  const startedAt = performance.now();
  const order = depthSortedOrder(lod.xyz, viewMatrix.elements);
  const sorted = {
    ...lod,
    xyz: reorderByOrder(lod.xyz, order),
    color: reorderByOrder(lod.color, order),
    opacity: reorderByOrder(lod.opacity, order),
    scale: reorderByOrder(lod.scale, order),
    rotation: Array.isArray(lod.rotation) ? reorderByOrder(lod.rotation, order) : lod.rotation,
  };
  return { lod: sorted, sortMs: performance.now() - startedAt };
}

async function ensureTransitionObject(count) {
  const key = String(count);
  if (state.transitionViewLock || state.autoSortView) {
    const lod = await getLod(key);
    if (state.transitionViewLock || lod.count <= AUTO_SORT_THRESHOLD) return ensureSortedTransitionObject(key);
  }
  let object = state.transitionObjects.get(key);
  if (!object) {
    const lod = await getLod(key);
    object = buildGaussianPoints(lod);
    object.visible = false;
    state.transitionObjects.set(key, object);
  }
  object.userData.lodCount = key;
  return object;
}

async function ensureSortedTransitionObject(count) {
  const key = `${state.preparedId}:${count}:${currentTransitionViewKey()}`;
  let object = state.sortedTransitionObjects.get(key);
  if (object) return object;

  const lod = await getLod(count);
  if (lod.count >= SORT_LOADING_THRESHOLD) {
    loadingOverlay.hidden = false;
    loadingMessage.textContent = `Sorting ${lod.count.toLocaleString()} splats for locked view...`;
    setStatus(`Sorting LOD ${count} for locked view...\nSplats: ${lod.count.toLocaleString()}`);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  const { lod: sortedLod, sortMs } = sortedLodForCurrentView(lod);
  object = buildGaussianPoints(sortedLod);
  object.visible = false;
  object.userData.kind = "gaussian";
  object.userData.lodCount = String(count);
  object.userData.sorted = true;
  object.userData.sortMs = sortMs;
  state.sortedTransitionObjects.set(key, object);
  state.sortedTransitionStats.set(String(count), { count: sortedLod.count, sortMs });
  loadingOverlay.hidden = !state.busy;
  return object;
}

function pruneStaleAutoSortedObjects() {
  if (state.transitionViewLock || !state.autoSortView) return;
  const suffix = `:${state.autoSortView.key}`;
  for (const [key, object] of state.sortedTransitionObjects.entries()) {
    if (!key.endsWith(suffix)) {
      disposeObject(object);
      state.sortedTransitionObjects.delete(key);
    }
  }
}

async function prepareAllSortedTransitionObjects() {
  const lodNames = (state.prepared?.lods ?? []).map((lod) => String(lod.name));
  if (!lodNames.length) return;
  const startedAt = performance.now();
  for (let index = 0; index < lodNames.length; index += 1) {
    const name = lodNames[index];
    loadingOverlay.hidden = false;
    loadingMessage.textContent = `Processing locked view ${index + 1}/${lodNames.length}: LOD ${name}`;
    setStatus(`Processing locked transition view...\nLOD ${index + 1}/${lodNames.length}: ${name}`);
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await ensureSortedTransitionObject(name);
  }
  const totalSplats = [...state.sortedTransitionStats.values()].reduce((sum, item) => sum + item.count, 0);
  const sortMs = [...state.sortedTransitionStats.values()].reduce((sum, item) => sum + item.sortMs, 0);
  setStatus([
    "Locked transition view is ready.",
    `Processed LODs: ${lodNames.length}`,
    `Total sorted splats: ${totalSplats.toLocaleString()}`,
    `Sorting time: ${sortMs.toFixed(0)}ms`,
    `Elapsed: ${((performance.now() - startedAt) / 1000).toFixed(1)}s`,
  ].join("\n"));
}

function transitionRadiusForSlider(t) {
  const viewerConfig = state.prepared?.viewer;
  if (!viewerConfig) return 0;
  const smoothT = t * t * (3 - 2 * t);
  return (1 - smoothT) * viewerConfig.far_radius + smoothT * viewerConfig.near_radius;
}

function cameraDistance() {
  return camera.position.distanceTo(controls.target);
}

function transitionProgressForRadius(radius) {
  const viewerConfig = state.prepared?.viewer;
  if (!viewerConfig) return 0;
  const far = Number(viewerConfig.far_radius) || 1;
  const near = Number(viewerConfig.near_radius) || 0;
  const range = Math.max(0.0001, far - near);
  return Math.max(0, Math.min(1, (far - radius) / range));
}

function setCameraDistance(radius) {
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() <= 1e-8) direction.set(1, 0.35, 1);
  direction.normalize();
  camera.position.copy(controls.target).addScaledVector(direction, radius);
  controls.update();
}

async function lockTransitionView() {
  setBusy(true, "Processing locked transition view...");
  camera.updateMatrixWorld(true);
  const position = camera.position.clone();
  const target = controls.target.clone();
  try {
    state.transitionViewLock = {
      position,
      target,
      key: makeTransitionViewKey(position, target),
      viewMatrix: camera.matrixWorldInverse.clone(),
    };
    clearSortedTransitionObjects();
    applyControlsLockState();
    updateTransitionLockUi();
    await prepareAllSortedTransitionObjects();
    await updateTransitionView();
  } catch (error) {
    state.transitionViewLock = null;
    clearSortedTransitionObjects();
    applyControlsLockState();
    setStatus(`Locked transition preparation failed:\n${error.message}`);
  } finally {
    setBusy(false);
  }
}

function unlockTransitionView() {
  state.transitionViewLock = null;
  clearSortedTransitionObjects();
  applyControlsLockState();
  updateTransitionLockUi();
  updateTransitionView().catch((error) => setStatus(`Transition update failed:\n${error.message}`));
}

function toggleTransitionViewLock() {
  if (state.transitionViewLock) unlockTransitionView();
  else lockTransitionView().catch((error) => setStatus(`Locked transition preparation failed:\n${error.message}`));
}

function updateTransitionLockUi() {
  if (!lockTransitionViewButton) return;
  const inTransition = modeSelect.value === "transition";
  setControlAvailability(lockTransitionViewButton, !state.busy && Boolean(state.prepared) && inTransition, "Only for Transition mode; depth-sorts splats for current camera.");
  lockTransitionViewButton.textContent = state.transitionViewLock ? "Unlock transition view" : "Lock transition view";
}

function updateManualVisibility() {
  setTransitionControlsEnabled(false);
  if (state.transitionViewLock) {
    state.transitionViewLock = null;
    clearSortedTransitionObjects();
    updateTransitionLockUi();
  }
  syncCameraLockForMode();
  const mode = modeSelect.value;
  if (state.meshObject) {
    state.meshObject.visible = mode === "mesh" || mode === "both";
    applyMeshOpacity(state.meshObject, MESH_OPACITY);
  }
  if (state.selectedGaussianObject) {
    state.selectedGaussianObject.visible = mode === "gaussian" || mode === "both";
    applyGaussianMaterial(state.selectedGaussianObject, Number(opacity.value));
  }
  for (const object of state.transitionObjects.values()) object.visible = false;
}

async function updateTransitionView({ syncCamera = false, syncSlider = true } = {}) {
  if (!state.prepared || !state.meshObject) return;
  syncCameraLockForMode();
  setTransitionControlsEnabled(true);
  const requestId = ++state.transitionRequestId;
  if (syncCamera) setCameraDistance(transitionRadiusForSlider(Number(transitionSlider.value)));
  const radius = cameraDistance();
  const t = transitionProgressForRadius(radius);
  if (syncSlider) transitionSlider.value = String(t);
  const style = transitionStyleSelect.value;
  if (!state.transitionViewLock) {
    camera.updateMatrixWorld();
    state.autoSortView = {
      key: makeTransitionViewKey(camera.position, controls.target),
      viewMatrix: camera.matrixWorldInverse.clone(),
    };
    pruneStaleAutoSortedObjects();
  }

  hideAllGaussians();

  if (t <= 0.0001) {
    state.meshObject.visible = true;
    applyMeshOpacity(state.meshObject, MESH_OPACITY);
    transitionValue.textContent = "0%";
    const lodNames = Object.keys(state.prepared.viewer.transition.lod_ranges ?? {});
    const hiddenCount = allGaussianObjects().length;
    setStatus([
      "Transition: 0%",
      `Camera distance: ${radius.toFixed(2)}`,
      state.transitionViewLock ? "View: locked, sorted splats ready on demand" : "View: auto-sorted for current camera",
      "mesh: 1.00",
      ...lodNames.map((name) => `${name}: 0.00`),
      `hidden gaussian objects: ${hiddenCount}`,
    ].join("\n"));
    return;
  }

  const weights = style === "additive"
    ? additiveTransitionWeights(radius, state.prepared.viewer.transition)
    : style === "dense"
      ? denseCutoverTransitionWeights(t, state.prepared.viewer.transition)
      : style === "detail"
        ? detailBuildTransitionWeights(t, state.prepared.viewer.transition)
        : transitionWeights(radius, state.prepared.viewer.transition);
  const visualMeshHold = 1 - smoothstep(0.68, 0.96, t);
  const meshOpacity = style === "additive"
    ? MESH_OPACITY
    : style === "detail"
      ? weights.mesh
      : t >= 0.995
        ? 0
        : Math.max(0.08, MESH_OPACITY * Math.max(weights.mesh, visualMeshHold));
  state.meshObject.visible = meshOpacity > 0.001;
  applyMeshOpacity(state.meshObject, meshOpacity);

  const activeLines = [
    `Transition: ${Math.round(t * 100)}%`,
    `Style: ${style === "additive" ? "mesh + added detail" : style === "dense" ? "dense LOD cutover" : style === "detail" ? "detail build-up" : "cross-fade"}`,
    `Camera distance: ${radius.toFixed(2)}`,
    state.transitionViewLock ? "View: locked, depth-sorted splats" : "View: auto-sorted for current camera",
    `mesh: ${meshOpacity.toFixed(2)}`,
  ];
  const activePromises = [];
  const activeObjects = [];
  for (const [count, weight] of Object.entries(weights.gaussian_lods)) {
    if (weight > 0.001) {
      activePromises.push(
        ensureTransitionObject(count).then((object) => {
          if (requestId !== state.transitionRequestId) return;
          object.visible = true;
          applyGaussianMaterial(object, Number(opacity.value) * weight);
          activeObjects.push({ count, weight, object });
        }),
      );
    } else {
      activeLines.push(`${count}: ${weight.toFixed(2)}`);
    }
  }
  await Promise.all(activePromises);
  if (requestId !== state.transitionRequestId) return;
  activeObjects.sort((a, b) => lodSortKey(a.count) - lodSortKey(b.count));
  for (const { count, weight, object } of activeObjects) {
    const sorted = object.userData.sorted ? "sorted" : "unsorted";
    const sortMs = object.userData.sortMs != null ? `, sort ${object.userData.sortMs.toFixed(0)}ms` : "";
    activeLines.push(`${count}: ${weight.toFixed(2)} (${sorted}, ${object.geometry.instanceCount.toLocaleString()} splats${sortMs})`);
  }
  for (const [count, object] of state.transitionObjects.entries()) {
    const weight = weights.gaussian_lods[count] ?? 0;
    if (weight <= 0.001) hideGaussianObject(object);
  }
  for (const [cacheKey, object] of state.sortedTransitionObjects.entries()) {
    const count = object.userData.lodCount;
    const weight = weights.gaussian_lods[count] ?? 0;
    if (weight <= 0.001 || !cacheKey.endsWith(`:${currentTransitionViewKey()}`)) hideGaussianObject(object);
  }
  transitionValue.textContent = `${Math.round(t * 100)}%`;
  setStatus(activeLines.join("\n"));
}

function scheduleTransitionFromCamera() {
  if (modeSelect.value !== "transition" || state.busy || !state.prepared || state.transitionViewLock) return;
  if (state.transitionCameraUpdateQueued) return;
  state.transitionCameraUpdateQueued = true;
  requestAnimationFrame(() => {
    state.transitionCameraUpdateQueued = false;
    updateTransitionView().catch((error) => setStatus(`Transition update failed:\n${error.message}`));
  });
}

function updateVisibility() {
  syncCameraLockForMode();
  if (modeSelect.value === "transition") {
    updateTransitionView().catch((error) => setStatus(`Transition update failed:\n${error.message}`));
  } else {
    state.transitionRequestId += 1;
    state.autoSortView = null;
    updateManualVisibility();
  }
}

async function updateMode() {
  if (state.busy) return;
  if (!state.preparedId) {
    updateVisibility();
    return;
  }
  if (modeSelect.value !== "transition" && (modeSelect.value === "gaussian" || modeSelect.value === "both") && !state.selectedGaussianObject) {
    await loadSelectedLod();
    return;
  }
  updateVisibility();
}

async function loadModels() {
  const data = await api("/api/models");
  modelSelect.innerHTML = "";
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    modelSelect.appendChild(option);
  }
  setStatus(`Found ${data.models.length} model option(s).`);
  updateControlAvailability();
}

async function loadTrainedGaussians({ preserveRepresentation = false } = {}) {
  const previousRepresentation = representationSelect.value;
  const data = await api("/api/trained-gaussians");
  const lodSetData = await api("/api/mesh2splat-lod-sets");
  trainedSelect.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = data.models.length ? "Auto match by mesh name" : "No single trained .ply found";
  trainedSelect.appendChild(none);
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    trainedSelect.appendChild(option);
  }
  if (preserveRepresentation) {
    representationSelect.value = previousRepresentation;
  } else if (lodSetData.sets.length > 0) {
    representationSelect.value = "mesh2splat_lods";
    setStatus(`Found ${lodSetData.sets.length} Mesh2Splat LOD set(s).`);
  } else if (data.models.length === 0) {
    representationSelect.value = "initialized";
  }
  updateControlAvailability();
}

async function refreshModelLists() {
  if (state.busy) return;
  setBusy(true, "Refreshing model and Gaussian lists...");
  try {
    await loadModels();
    await loadTrainedGaussians({ preserveRepresentation: true });
    setStatus("Lists refreshed.");
  } catch (error) {
    setStatus(`Refresh failed:\n${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function prepareSelectedModel() {
  if (state.busy) return;
  setBusy(true, "Preparing mesh and Gaussian LODs...");
  try {
    const prepared = await api("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelSelect.value,
        representation: representationSelect.value,
        trained_ply_id: trainedSelect.value || null,
        lod_counts: representationSelect.value === "trained" ? DEFAULT_TRAINED_LOD_COUNTS : undefined,
      }),
    });
    setStatus("Loading mesh and viewer objects...");
    await applyPreparedModel(prepared);
  } catch (error) {
    setStatus(`Failed:\n${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function applyPreparedModel(prepared) {
  setStatus("Clearing previous viewer data...");
  setMeshStatus("");
  state.transitionViewLock = null;
  state.cameraLock.enabled = false;
  state.cameraLock.automatic = false;
  state.cameraLock.userUnlockedAuto = false;
  applyControlsLockState();
  state.preparedId = prepared.id;
  state.prepared = prepared;
  state.lodCache.clear();
  disposeObject(state.meshObject);
  disposeObject(state.selectedGaussianObject);
  clearTransitionObjects();
  updateTransitionLockUi();
  updateCameraLockUi();
  setStatus("Loading source mesh...");
  state.meshObject = await buildSceneMesh(prepared.mesh);
  scene.add(state.meshObject);

  lodSelect.innerHTML = "";
  for (const lod of prepared.lods) {
    const option = document.createElement("option");
    option.value = lod.name;
    option.textContent = `${lod.name.toLocaleString?.() ?? lod.name} target / ${lod.count.toLocaleString()} loaded`;
    lodSelect.appendChild(option);
  }
  if (prepared.representation === "trained" && lodSelect.options.length > 0) {
    lodSelect.selectedIndex = lodSelect.options.length - 1;
  }

  transitionSlider.value = "0";
  transitionValue.textContent = "0%";
  if (modeSelect.value === "transition") {
    setStatus("Preparing transition view...");
    await updateTransitionView();
  } else {
    setStatus("Loading selected Gaussian LOD...");
    await loadSelectedLod();
  }
  if (state.meshObject?.userData?.loadFailed) setStatus(state.meshObject.userData.message);
  syncCameraLockForMode();
}

async function convertSelectedWithMesh2Splat() {
  if (state.busy) return;
  setBusy(true, "Running Mesh2Splat conversion...");
  setStatus("Running Mesh2Splat conversion...");
  try {
    const prepared = await api("/api/convert-mesh2splat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelSelect.value,
        density: Number(mesh2splatDensity.value),
      }),
    });
    representationSelect.value = "trained";
    await loadTrainedGaussians({ preserveRepresentation: true });
    representationSelect.value = "trained";
    const match = [...trainedSelect.options].find((option) => option.textContent && prepared.gaussian_source?.endsWith(option.textContent));
    if (match) trainedSelect.value = match.value;
    await applyPreparedModel(prepared);
    setStatus([
      "Mesh2Splat conversion complete.",
      `PLY: ${prepared.conversion?.output_ply ?? prepared.gaussian_source}`,
      `GLB: ${prepared.conversion?.glb_mesh ?? "input was already GLB"}`,
      `Prepared ${prepared.lods.length} LOD level(s).`,
    ].join("\n"));
  } catch (error) {
    setStatus(`Mesh2Splat conversion failed:\n${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function loadSelectedLod() {
  if (state.busy && !state.prepared) return;
  if (!state.preparedId || !lodSelect.value) return;
  if (modeSelect.value === "transition") {
    hideGaussianObject(state.selectedGaussianObject);
    await updateTransitionView();
    return;
  }
  setStatus(`Loading selected LOD ${lodSelect.value}...`);
  const lod = await getLod(lodSelect.value);
  disposeObject(state.selectedGaussianObject);
  state.selectedGaussianObject = buildGaussianPoints(lod);
  updateVisibility();
}

async function uploadModel() {
  if (state.busy) return;
  const file = uploadInput.files?.[0];
  if (!file) return;
  setBusy(true, `Uploading ${file.name}...`);
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api("/api/upload", { method: "POST", body: form });
    await loadModels();
    modelSelect.value = result.model.id;
    setStatus(`Uploaded ${result.model.name}.`);
  } catch (error) {
    setStatus(`Upload failed:\n${error.message}`);
  } finally {
    setBusy(false);
  }
}

function updatePointMaterial() {
  if (state.busy) return;
  if (state.selectedGaussianObject) applyGaussianMaterial(state.selectedGaussianObject, Number(opacity.value));
  for (const object of state.transitionObjects.values()) {
    applyGaussianMaterial(object, object.opacityMultiplier);
  }
  for (const object of state.sortedTransitionObjects.values()) {
    applyGaussianMaterial(object, object.opacityMultiplier);
  }
  updateVisibility();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  drawRawGaussianLayer();
}

window.addEventListener("resize", resize);
refreshModelsButton.addEventListener("click", refreshModelLists);
prepareButton.addEventListener("click", prepareSelectedModel);
convertMesh2SplatButton.addEventListener("click", convertSelectedWithMesh2Splat);
uploadInput.addEventListener("change", uploadModel);
lodSelect.addEventListener("change", loadSelectedLod);
modelSelect.addEventListener("change", updateControlAvailability);
representationSelect.addEventListener("change", updateControlAvailability);
modeSelect.addEventListener("change", () => updateMode().catch((error) => setStatus(`Mode update failed:\n${error.message}`)));
transitionStyleSelect.addEventListener("change", updateVisibility);
backgroundSelect.addEventListener("change", applyBackgroundTheme);
transitionSlider.addEventListener("input", () => updateTransitionView({ syncCamera: true, syncSlider: false }).catch((error) => setStatus(`Transition update failed:\n${error.message}`)));
lockTransitionViewButton.addEventListener("click", toggleTransitionViewLock);
lockCameraButton.addEventListener("click", toggleCameraLock);
pointSize.addEventListener("input", updatePointMaterial);
opacity.addEventListener("input", updatePointMaterial);
gaussianYOffset.addEventListener("input", updatePointMaterial);
gaussianScale.addEventListener("input", updatePointMaterial);
controls.addEventListener("change", scheduleTransitionFromCamera);

resize();
applyBackgroundTheme();
updateTransitionLockUi();
updateCameraLockUi();
animate();
loadModels().catch((error) => setStatus(`Failed to load models:\n${error.message}`));
loadTrainedGaussians().catch((error) => setStatus(`Failed to load trained Gaussian files:\n${error.message}`));

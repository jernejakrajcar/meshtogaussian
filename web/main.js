import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const modelSelect = document.querySelector("#modelSelect");
const uploadInput = document.querySelector("#uploadInput");
const prepareButton = document.querySelector("#prepareButton");
const representationSelect = document.querySelector("#representationSelect");
const trainedSelect = document.querySelector("#trainedSelect");
const modeSelect = document.querySelector("#modeSelect");
const transitionSlider = document.querySelector("#transitionSlider");
const transitionValue = document.querySelector("#transitionValue");
const lodSelect = document.querySelector("#lodSelect");
const pointSize = document.querySelector("#pointSize");
const opacity = document.querySelector("#opacity");
const statusBox = document.querySelector("#status");
const viewer = document.querySelector("#viewer");

const state = {
  preparedId: null,
  prepared: null,
  meshObject: null,
  selectedGaussianObject: null,
  transitionObjects: new Map(),
  lodCache: new Map(),
  transitionRequestId: 0,
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2.2, 1.3, 2.6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, 2.2);
light.position.set(2.0, 4.0, 3.0);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
scene.add(new THREE.GridHelper(2.4, 12, 0x46515c, 0x252c33));

function setStatus(message) {
  statusBox.textContent = message;
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

function resize() {
  const rect = viewer.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
}

function disposeObject(object) {
  if (!object) return;
  scene.remove(object);
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    child.material?.dispose?.();
  });
}

function allGaussianObjects() {
  const objects = [];
  scene.traverse((object) => {
    if (object.userData?.kind === "gaussian") objects.push(object);
  });
  return objects;
}

function clearTransitionObjects() {
  for (const object of state.transitionObjects.values()) disposeObject(object);
  state.transitionObjects.clear();
}

function buildMesh(mesh) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(mesh.colors.flat(), 3));
  geometry.setIndex(mesh.faces.flat());
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.05,
    roughness: 0.72,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.82,
  });
  return new THREE.Mesh(geometry, material);
}

function buildGaussianPoints(lod) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(lod.xyz.flat(), 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(lod.color.flat(), 3));
  geometry.setAttribute("splatScale", new THREE.Float32BufferAttribute(lod.scale, 1));
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexColors: true,
    uniforms: {
      opacityMultiplier: { value: Number(opacity.value) },
      pointSizeMultiplier: { value: Number(pointSize.value) * 950.0 },
    },
    vertexShader: `
      attribute float splatScale;
      uniform float pointSizeMultiplier;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        float size = max(1.0, pointSizeMultiplier * splatScale / max(0.05, -mvPosition.z));
        gl_PointSize = size;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform float opacityMultiplier;
      varying vec3 vColor;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(p, p);
        if (r2 > 1.0) discard;
        float alpha = exp(-3.0 * r2) * opacityMultiplier;
        gl_FragColor = vec4(vColor, alpha);
      }
    `,
  });
  const points = new THREE.Points(geometry, material);
  points.userData.kind = "gaussian";
  return points;
}

function hideGaussianObject(object) {
  if (!object) return;
  object.visible = false;
  if (object.material.uniforms?.opacityMultiplier) object.material.uniforms.opacityMultiplier.value = 0;
  else object.material.opacity = 0;
  object.material.needsUpdate = true;
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

async function ensureTransitionObject(count) {
  const key = String(count);
  let object = state.transitionObjects.get(key);
  if (!object) {
    const lod = await getLod(key);
    object = buildGaussianPoints(lod);
    object.visible = false;
    scene.add(object);
    state.transitionObjects.set(key, object);
  }
  return object;
}

function setCameraByTransition(t) {
  const viewerConfig = state.prepared?.viewer;
  if (!viewerConfig) return 0;
  const smoothT = t * t * (3 - 2 * t);
  const radius = (1 - smoothT) * viewerConfig.far_radius + smoothT * viewerConfig.near_radius;
  const az = THREE.MathUtils.degToRad(viewerConfig.azimuth_degrees);
  const elev = THREE.MathUtils.degToRad(viewerConfig.elevation_degrees);
  camera.position.set(
    radius * Math.cos(elev) * Math.sin(az),
    radius * Math.sin(elev),
    radius * Math.cos(elev) * Math.cos(az),
  );
  controls.target.set(0, 0, 0);
  controls.update();
  return radius;
}

function updateManualVisibility() {
  const mode = modeSelect.value;
  if (state.meshObject) {
    state.meshObject.visible = mode === "mesh" || mode === "both";
    state.meshObject.material.opacity = 0.82;
  }
  if (state.selectedGaussianObject) {
    state.selectedGaussianObject.visible = mode === "gaussian" || mode === "both";
    state.selectedGaussianObject.material.uniforms.opacityMultiplier.value = Number(opacity.value);
  }
  for (const object of state.transitionObjects.values()) object.visible = false;
}

async function updateTransitionView() {
  if (!state.prepared || !state.meshObject) return;
  const requestId = ++state.transitionRequestId;
  const t = Number(transitionSlider.value);
  const radius = setCameraByTransition(t);

  hideAllGaussians();

  if (t <= 0.0001) {
    state.meshObject.visible = true;
    state.meshObject.material.opacity = 0.82;
    transitionValue.textContent = "0%";
    const lodNames = Object.keys(state.prepared.viewer.transition.lod_ranges ?? {});
    const hiddenCount = allGaussianObjects().length;
    setStatus([
      "Transition: 0%",
      `Camera distance: ${radius.toFixed(2)}`,
      "mesh: 1.00",
      ...lodNames.map((name) => `${name}: 0.00`),
      `hidden gaussian objects: ${hiddenCount}`,
    ].join("\n"));
    return;
  }

  const weights = transitionWeights(radius, state.prepared.viewer.transition);
  state.meshObject.visible = weights.mesh > 0.01;
  state.meshObject.material.opacity = Math.max(0.04, 0.82 * weights.mesh);

  const activeLines = [`Transition: ${Math.round(t * 100)}%`, `Camera distance: ${radius.toFixed(2)}`, `mesh: ${weights.mesh.toFixed(2)}`];
  const activePromises = [];
  for (const [count, weight] of Object.entries(weights.gaussian_lods)) {
    activeLines.push(`${count}: ${weight.toFixed(2)}`);
    if (weight > 0.001) {
      activePromises.push(
        ensureTransitionObject(count).then((object) => {
          if (requestId !== state.transitionRequestId) return;
          object.visible = true;
          object.material.uniforms.opacityMultiplier.value = Number(opacity.value) * weight;
          object.material.uniforms.pointSizeMultiplier.value = Number(pointSize.value) * 950.0;
          object.material.needsUpdate = true;
        }),
      );
    }
  }
  await Promise.all(activePromises);
  if (requestId !== state.transitionRequestId) return;
  for (const [count, object] of state.transitionObjects.entries()) {
    const weight = weights.gaussian_lods[count] ?? 0;
    if (weight <= 0.001) hideGaussianObject(object);
  }
  transitionValue.textContent = `${Math.round(t * 100)}%`;
  setStatus(activeLines.join("\n"));
}

function updateVisibility() {
  if (modeSelect.value === "transition") {
    updateTransitionView().catch((error) => setStatus(`Transition update failed:\n${error.message}`));
  } else {
    updateManualVisibility();
  }
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
}

async function loadTrainedGaussians() {
  const data = await api("/api/trained-gaussians");
  trainedSelect.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = data.models.length ? "Auto match by mesh name" : "No trained .ply found";
  trainedSelect.appendChild(none);
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    trainedSelect.appendChild(option);
  }
  if (data.models.length === 0) representationSelect.value = "initialized";
}

async function prepareSelectedModel() {
  prepareButton.disabled = true;
  setStatus("Preparing mesh and Gaussian LODs...");
  try {
    const prepared = await api("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelSelect.value,
        representation: representationSelect.value,
        trained_ply_id: trainedSelect.value || null,
      }),
    });

    state.preparedId = prepared.id;
    state.prepared = prepared;
    state.lodCache.clear();
    disposeObject(state.meshObject);
    disposeObject(state.selectedGaussianObject);
    clearTransitionObjects();
    state.meshObject = buildMesh(prepared.mesh);
    scene.add(state.meshObject);

    lodSelect.innerHTML = "";
    for (const lod of prepared.lods) {
      const option = document.createElement("option");
      option.value = lod.count;
      option.textContent = `${lod.count.toLocaleString()} Gaussians`;
      lodSelect.appendChild(option);
    }

    transitionSlider.value = "0";
    transitionValue.textContent = "0%";
    if (modeSelect.value === "transition") {
      await updateTransitionView();
    } else {
      await loadSelectedLod();
    }
  } catch (error) {
    setStatus(`Failed:\n${error.message}`);
  } finally {
    prepareButton.disabled = false;
  }
}

async function loadSelectedLod() {
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
  scene.add(state.selectedGaussianObject);
  updateVisibility();
}

async function uploadModel() {
  const file = uploadInput.files?.[0];
  if (!file) return;
  setStatus(`Uploading ${file.name}...`);
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api("/api/upload", { method: "POST", body: form });
    await loadModels();
    modelSelect.value = result.model.id;
    setStatus(`Uploaded ${result.model.name}.`);
  } catch (error) {
    setStatus(`Upload failed:\n${error.message}`);
  }
}

function updatePointMaterial() {
  if (state.selectedGaussianObject) {
    state.selectedGaussianObject.material.uniforms.pointSizeMultiplier.value = Number(pointSize.value) * 950.0;
    state.selectedGaussianObject.material.uniforms.opacityMultiplier.value = Number(opacity.value);
    state.selectedGaussianObject.material.needsUpdate = true;
  }
  for (const object of state.transitionObjects.values()) {
    object.material.uniforms.pointSizeMultiplier.value = Number(pointSize.value) * 950.0;
    object.material.needsUpdate = true;
  }
  updateVisibility();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", resize);
prepareButton.addEventListener("click", prepareSelectedModel);
uploadInput.addEventListener("change", uploadModel);
lodSelect.addEventListener("change", loadSelectedLod);
modeSelect.addEventListener("change", updateVisibility);
transitionSlider.addEventListener("input", () => updateTransitionView().catch((error) => setStatus(`Transition update failed:\n${error.message}`)));
pointSize.addEventListener("input", updatePointMaterial);
opacity.addEventListener("input", updatePointMaterial);

resize();
animate();
loadModels().catch((error) => setStatus(`Failed to load models:\n${error.message}`));
loadTrainedGaussians().catch((error) => setStatus(`Failed to load trained Gaussian files:\n${error.message}`));

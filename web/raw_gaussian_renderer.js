function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create raw Gaussian shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Raw Gaussian shader compilation failed: ${info}`);
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create raw Gaussian program.");
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Raw Gaussian program linking failed: ${info}`);
  }
  return program;
}

const VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aPosition;
layout(location = 2) in vec4 aColor;
layout(location = 3) in vec3 aScale;
layout(location = 4) in vec4 aQuat;
layout(location = 5) in float aRank;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform float uPointScale;
uniform float uGaussianScale;
uniform float uYOffset;
uniform float uMaxPointSize;
uniform float uMaxGaussianPointSize;
uniform vec2 uViewport;
uniform float uOpacityMultiplier;
uniform float uRevealEnabled;
uniform float uRevealFullCount;
uniform float uRevealPartialCount;
uniform float uRevealMix;

out vec4 vColor;
out vec2 vDxPixels;
out mat2 vInvCov2;

mat3 quatToMat3(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;

  return mat3(
    1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),
    2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),
    2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)
  );
}

void main() {
  float reveal = 1.0;
  if (uRevealEnabled > 0.5) {
    if (aRank >= uRevealPartialCount) {
      reveal = 0.0;
    } else if (aRank >= uRevealFullCount) {
      reveal = uRevealMix;
    }
  }
  if (reveal <= 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vDxPixels = vec2(0.0);
    vInvCov2 = mat2(1.0);
    vColor = vec4(0.0);
    return;
  }

  vec4 viewPos = uViewMatrix * vec4(aPosition + vec3(0.0, uYOffset, 0.0), 1.0);
  float depth = -viewPos.z;

  // A screen-aligned quad cannot be safely clipped through the camera plane.
  // Reject near/behind-camera splats instead of projecting them into huge streaks.
  if (depth <= 0.05) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vDxPixels = vec2(0.0);
    vInvCov2 = mat2(1.0);
    vColor = vec4(0.0);
    return;
  }

  vec3 safeScale = max(aScale * uGaussianScale, vec3(1e-6));
  mat3 R = quatToMat3(normalize(aQuat));
  mat3 S2 = mat3(
    safeScale.x * safeScale.x, 0.0, 0.0,
    0.0, safeScale.y * safeScale.y, 0.0,
    0.0, 0.0, safeScale.z * safeScale.z
  );
  mat3 sigmaWorld = R * S2 * transpose(R);
  mat3 viewRot = mat3(uViewMatrix);
  mat3 sigmaView = viewRot * sigmaWorld * transpose(viewRot);

  float z = depth;
  float x = viewPos.x;
  float y = viewPos.y;
  float invZ = 1.0 / z;
  float invZ2 = invZ * invZ;

  float fx = uProjectionMatrix[0][0];
  float fy = uProjectionMatrix[1][1];
  float halfW = 0.5 * uViewport.x;
  float halfH = 0.5 * uViewport.y;

  float j00 = halfW * fx * invZ;
  float j11 = halfH * fy * invZ;
  float j02 = halfW * -fx * x * invZ2;
  float j12 = halfH * -fy * y * invZ2;
  mat3 jpix = mat3(
    j00, 0.0, 0.0,
    0.0, j11, 0.0,
    j02, j12, 0.0
  );

  mat3 sigmaScreen3 = (jpix * sigmaView) * transpose(jpix);
  mat2 sigma2 = mat2(
    sigmaScreen3[0][0], sigmaScreen3[0][1],
    sigmaScreen3[1][0], sigmaScreen3[1][1]
  );

  float trace0 = sigma2[0][0] + sigma2[1][1];
  float det0 = sigma2[0][0] * sigma2[1][1] - sigma2[0][1] * sigma2[1][0];
  float disc0 = max(trace0 * trace0 - 4.0 * det0, 0.0);
  float sdisc0 = sqrt(disc0);
  float lambdaMin0 = 0.5 * (trace0 - sdisc0);
  float lambdaFloor = 0.25;
  sigma2 += mat2(max(0.0, lambdaFloor - lambdaMin0));
  sigma2 += mat2(max(1e-4, 1e-6 * (sigma2[0][0] + sigma2[1][1])));
  vInvCov2 = inverse(sigma2);

  float trace = sigma2[0][0] + sigma2[1][1];
  float detv = sigma2[0][0] * sigma2[1][1] - sigma2[0][1] * sigma2[1][0];
  float disc = max(trace * trace - 4.0 * detv, 0.0);
  float lambdaMax = 0.5 * (trace + sqrt(disc));
  float sigmaMax = sqrt(max(lambdaMax, 1e-8));
  float size = (2.0 * 3.0 * sigmaMax) * uPointScale;
  float clampedSize = min(min(size, uMaxPointSize), uMaxGaussianPointSize);
  vec2 offsetPixels = aCorner * (0.5 * clampedSize);

  vec4 clip = uProjectionMatrix * viewPos;
  clip.xy += (offsetPixels / uViewport) * 2.0 * clip.w;
  gl_Position = clip;
  vDxPixels = offsetPixels;

  float alpha = clamp(aColor.a * uOpacityMultiplier * reveal, 0.0, 1.0);
  vColor = vec4(aColor.rgb * alpha, alpha);
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vDxPixels;
in mat2 vInvCov2;

out vec4 outColor;

void main() {
  float quad = dot(vDxPixels, vInvCov2 * vDxPixels);
  if (quad > 9.0) discard;

  float g = exp(-0.5 * quad);
  vec4 col = vec4(vColor.rgb * g, vColor.a * g);
  if (col.a <= 0.00392157) discard;
  outColor = col;
}
`;

function flattenVec3(values, count, fallback = 0) {
  if (ArrayBuffer.isView(values) && values.length === count * 3) {
    return values instanceof Float32Array ? values : new Float32Array(values);
  }
  const result = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const value = values?.[i];
    if (Array.isArray(value)) {
      result[i * 3 + 0] = Number(value[0]) || fallback;
      result[i * 3 + 1] = Number(value[1]) || fallback;
      result[i * 3 + 2] = Number(value[2]) || fallback;
    } else {
      const scalar = Number(value) || fallback;
      result[i * 3 + 0] = scalar;
      result[i * 3 + 1] = scalar;
      result[i * 3 + 2] = scalar;
    }
  }
  return result;
}

function flattenColors(colors, opacity, count) {
  const result = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    if (ArrayBuffer.isView(colors)) {
      result[i * 4 + 0] = colors[i * 3];
      result[i * 4 + 1] = colors[i * 3 + 1];
      result[i * 4 + 2] = colors[i * 3 + 2];
    } else {
      const color = colors?.[i] ?? [0.85, 0.68, 0.36];
      result[i * 4 + 0] = Number(color[0]) || 0;
      result[i * 4 + 1] = Number(color[1]) || 0;
      result[i * 4 + 2] = Number(color[2]) || 0;
    }
    result[i * 4 + 3] = Math.max(0, Math.min(1, Number(opacity?.[i]) || 0));
  }
  return result;
}

export function graphDecoToShaderQuat(value) {
  if (!Array.isArray(value) || value.length < 4) return [0, 0, 0, 1];
  return [Number(value[1]) || 0, Number(value[2]) || 0, Number(value[3]) || 0, Number(value[0]) || 1];
}

function flattenRotations(rotations, count) {
  const result = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    if (ArrayBuffer.isView(rotations)) {
      result[i * 4 + 0] = rotations[i * 4 + 1];
      result[i * 4 + 1] = rotations[i * 4 + 2];
      result[i * 4 + 2] = rotations[i * 4 + 3];
      result[i * 4 + 3] = rotations[i * 4];
    } else {
      const quat = graphDecoToShaderQuat(rotations?.[i]);
      result[i * 4 + 0] = quat[0];
      result[i * 4 + 1] = quat[1];
      result[i * 4 + 2] = quat[2];
      result[i * 4 + 3] = quat[3];
    }
  }
  return result;
}

function flattenRanks(ranks, count) {
  if (ArrayBuffer.isView(ranks) && ranks.length === count) {
    return ranks instanceof Float32Array ? ranks : new Float32Array(ranks);
  }
  const result = new Float32Array(count);
  for (let i = 0; i < count; i += 1) result[i] = i;
  return result;
}

export function detailRevealAlpha(rank, fullCount, partialCount, mix) {
  if (rank < fullCount) return 1;
  if (rank < partialCount) return mix;
  return 0;
}

function createLayer(gl, lod) {
  const count = Number(lod.count) || (ArrayBuffer.isView(lod.xyz) ? lod.xyz.length / 3 : lod.xyz.length);
  const vao = gl.createVertexArray();
  const buffers = {
    corners: gl.createBuffer(),
    indices: gl.createBuffer(),
    positions: gl.createBuffer(),
    colors: gl.createBuffer(),
    scales: gl.createBuffer(),
    rotations: gl.createBuffer(),
    ranks: gl.createBuffer(),
  };
  if (!vao || Object.values(buffers).some((buffer) => !buffer)) {
    throw new Error("Failed to create raw Gaussian buffers.");
  }

  gl.bindVertexArray(vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.corners);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.positions);
  gl.bufferData(gl.ARRAY_BUFFER, flattenVec3(lod.xyz, count), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(1, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colors);
  gl.bufferData(gl.ARRAY_BUFFER, flattenColors(lod.color, lod.opacity, count), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(2, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.scales);
  gl.bufferData(gl.ARRAY_BUFFER, flattenVec3(lod.scale, count, 0.001), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 3, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(3, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.rotations);
  gl.bufferData(gl.ARRAY_BUFFER, flattenRotations(lod.rotation, count), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(4);
  gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(4, 1);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.ranks);
  gl.bufferData(gl.ARRAY_BUFFER, flattenRanks(lod.rank, count), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(5);
  gl.vertexAttribPointer(5, 1, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(5, 1);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

  return {
    visible: true,
    opacityMultiplier: 1,
    reveal: null,
    geometry: { instanceCount: count },
    userData: { kind: "gaussian" },
    dispose() {
      gl.deleteVertexArray(vao);
      for (const buffer of Object.values(buffers)) gl.deleteBuffer(buffer);
    },
    _raw: { vao, count },
  };
}

export function createRawGaussianRenderer(canvas, options = {}) {
  const gl = canvas.getContext("webgl2", { alpha: true, antialias: false, depth: true, premultipliedAlpha: true });
  if (!gl) throw new Error("WebGL2 is required for raw Gaussian rendering.");
  const maxPointSize = options.maxPointSize ?? 2048;
  const maxGaussianPointSize = Math.min(options.maxGaussianPointSize ?? 2048, maxPointSize);
  const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  const uniforms = {
    viewMatrix: gl.getUniformLocation(program, "uViewMatrix"),
    projectionMatrix: gl.getUniformLocation(program, "uProjectionMatrix"),
    pointScale: gl.getUniformLocation(program, "uPointScale"),
    gaussianScale: gl.getUniformLocation(program, "uGaussianScale"),
    yOffset: gl.getUniformLocation(program, "uYOffset"),
    maxPointSize: gl.getUniformLocation(program, "uMaxPointSize"),
    maxGaussianPointSize: gl.getUniformLocation(program, "uMaxGaussianPointSize"),
    viewport: gl.getUniformLocation(program, "uViewport"),
    opacityMultiplier: gl.getUniformLocation(program, "uOpacityMultiplier"),
    revealEnabled: gl.getUniformLocation(program, "uRevealEnabled"),
    revealFullCount: gl.getUniformLocation(program, "uRevealFullCount"),
    revealPartialCount: gl.getUniformLocation(program, "uRevealPartialCount"),
    revealMix: gl.getUniformLocation(program, "uRevealMix"),
  };
  const layers = new Set();

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);

  function uploadScene(lod) {
    const layer = createLayer(gl, lod);
    layers.add(layer);
    return layer;
  }

  function disposeLayer(layer) {
    if (!layer || !layers.has(layer)) return;
    layers.delete(layer);
    layer.dispose();
  }

  function setSize(width, height, pixelRatio = 1) {
    const nextWidth = Math.max(1, Math.floor(width * pixelRatio));
    const nextHeight = Math.max(1, Math.floor(height * pixelRatio));
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    gl.viewport(0, 0, nextWidth, nextHeight);
  }

  function draw({ camera, viewport, opacity = 1, pointScale = 1, gaussianScale = 1, yOffset = 0, clear = true } = {}) {
    const width = viewport?.[0] ?? canvas.width;
    const height = viewport?.[1] ?? canvas.height;
    gl.viewport(0, 0, width, height);
    if (clear) gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!camera) return;
    gl.useProgram(program);
    gl.uniformMatrix4fv(uniforms.viewMatrix, false, camera.matrixWorldInverse.elements);
    gl.uniformMatrix4fv(uniforms.projectionMatrix, false, camera.projectionMatrix.elements);
    gl.uniform1f(uniforms.pointScale, pointScale);
    gl.uniform1f(uniforms.gaussianScale, gaussianScale);
    gl.uniform1f(uniforms.yOffset, yOffset);
    gl.uniform1f(uniforms.maxPointSize, maxPointSize);
    gl.uniform1f(uniforms.maxGaussianPointSize, maxGaussianPointSize);
    gl.uniform2f(uniforms.viewport, width, height);

    for (const layer of layers) {
      if (!layer.visible || layer.opacityMultiplier <= 0) continue;
      gl.uniform1f(uniforms.opacityMultiplier, opacity * layer.opacityMultiplier);
      gl.uniform1f(uniforms.revealEnabled, layer.reveal ? 1 : 0);
      gl.uniform1f(uniforms.revealFullCount, layer.reveal?.fullCount ?? layer._raw.count);
      gl.uniform1f(uniforms.revealPartialCount, layer.reveal?.partialCount ?? layer._raw.count);
      gl.uniform1f(uniforms.revealMix, layer.reveal?.mix ?? 0);
      gl.bindVertexArray(layer._raw.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, layer._raw.count);
    }
    gl.bindVertexArray(null);
  }

  function dispose() {
    for (const layer of [...layers]) disposeLayer(layer);
    gl.deleteProgram(program);
  }

  return {
    uploadScene,
    uploadSorted: uploadScene,
    disposeLayer,
    setSize,
    draw,
    dispose,
    get limits() {
      return { maxPointSize, maxGaussianPointSize };
    },
  };
}
